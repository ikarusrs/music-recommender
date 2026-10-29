// music.h -- everything music-specific, built on core.h.
//
// Flow: library CSV -> songs, catalog CSV streamed once for audio features
// -> min-max normalise -> vectors -> K-d tree (in core.h). Search results
// are dataset indices, mapped back to songs by MusicRecommender.

#ifndef MUSIC_H
#define MUSIC_H

#include "core.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <chrono>
#include <cmath>
#include <algorithm>

// Song metadata

// Everything we display about a song. Plain values only: vectors live in the
// KDataset and every song maps to one index of that dataset.
struct SongMeta {
    std::string trackId;    // Spotify track id (the matching key)
    std::string trackName;
    std::string artist;
    std::string album;
    std::string country;    // ISO country code(s)
};

// A song plus its dataset index (used for search-by-name listings).
struct MatchedSong {
    SongMeta meta;
    std::size_t vectorIndex = 0;
};

// A query-only song: metadata plus raw features in the corpus axis order.
// Loaded from a separate inline-feature CSV when the user wants to ask
// "what in the big library is similar to this favourite?".
struct QuerySong {
    SongMeta meta;
    std::vector<double> features;
};

// CSV parsing

// Parses one CSV record starting at text[pos] and moves pos past it.
// Handles quoted fields (commas, newlines and "" inside quotes), \n and \r\n
// line ends. Bad quoting (junk after a closing quote, or a quote that never
// closes) sets *malformedOut if it's given. Returns false at end of text.
inline bool nextCsvRecord(const std::string& text, std::size_t& pos,
                          char delim, std::vector<std::string>& fields,
                          bool* malformedOut = nullptr) {
    fields.clear();
    if (pos >= text.size()) return false;
    enum class QuoteState { Normal, InQuotes, JustClosed };
    QuoteState state = QuoteState::Normal;
    bool malformed = false;
    std::string current;
    std::size_t i = pos;
    for (; i < text.size(); ++i) {
        char c = text[i];
        if (state == QuoteState::InQuotes) {
            if (c == '"') {
                state = QuoteState::JustClosed;    // tentative closing quote
            } else {
                current += c;
            }
        } else if (state == QuoteState::JustClosed) {
            if (c == '"') {
                current += '"';                    // "" -> escaped literal quote
                state = QuoteState::InQuotes;
            } else if (c == delim) {
                fields.push_back(current);
                current.clear();
                state = QuoteState::Normal;
            } else if (c == '\n') {
                ++i;
                break;                             // record end
            } else if (c == '\r') {
                if (i + 1 < text.size() && text[i + 1] == '\n') ++i;
                ++i;
                break;                             // record end
            } else {
                malformed = true;                  // garbage after closing quote
                current += c;                      // carry on to the real boundary
                state = QuoteState::Normal;
            }
        } else {                                   // Normal
            if (c == '"') {
                state = QuoteState::InQuotes;      // opening quote
            } else if (c == delim) {
                fields.push_back(current);
                current.clear();
            } else if (c == '\n') {
                ++i;
                break;                             // record end
            } else if (c == '\r') {
                if (i + 1 < text.size() && text[i + 1] == '\n') ++i;
                ++i;
                break;                             // record end
            } else {
                current += c;                      // plain field content
            }
        }
    }
    if (state == QuoteState::InQuotes) malformed = true;   // unterminated field
    fields.push_back(current);
    pos = i;
    if (malformedOut != nullptr) *malformedOut = malformed;
    return true;
}

// Streaming CSV reader. Reads the file in ~1 MiB chunks because the feature
// catalog is too big (234 MB) to load whole. When a record runs past the end
// of the chunk, it refills the buffer and parses that record again from the
// start. A max record size stops a broken file from growing the buffer forever.
class CsvStream {
public:
    explicit CsvStream(std::ifstream& src) : src_(src) {
        chunk_.resize(kChunk);
    }

    // Parses the next real record (blank records are skipped, matching the
    // old reader). Returns false at clean end of input OR on failure; the
    // reason is in error() afterwards (empty means "just end of input").
    // Opens nothing: the caller owns and opens the ifstream.
    bool next(std::vector<std::string>& fields) {
        while (true) {
            bool malformed = false;
            std::size_t pos = readPos_;
            if (nextCsvRecord(buffer_, pos, ',', fields, &malformed)) {
                bool terminated = pos > readPos_ &&
                    (buffer_[pos - 1] == '\n' || buffer_[pos - 1] == '\r');
                bool finalRecord = eof_ && pos >= buffer_.size();
                if (terminated || finalRecord) {
                    if (malformed) {
                        error_ = "malformed CSV: unterminated quoted field or "
                                 "characters after a closing quote";
                        return false;              // stop with a clear error
                    }
                    readPos_ = pos;
                    if (readPos_ > kChunk) {       // reclaim consumed prefix
                        buffer_.erase(0, readPos_);
                        readPos_ = 0;
                    }
                    if (!blankRecord(fields)) return true;
                    continue;                      // skip blank records
                }
                // NOT terminated: continue reading below and re-parse.
            } else if (eof_) {
                return false;                      // clean end of input
            }
            if (eof_) continue;                    // parse the final record first
            if (buffer_.size() - readPos_ > kMaxRecord) {
                error_ = "malformed CSV: single record exceeds " +
                         std::to_string(kMaxRecord) + " bytes (no terminator)";
                return false;                      // never resume mid-record
            }
            src_.read(chunk_.data(), static_cast<std::streamsize>(chunk_.size()));
            std::streamsize got = src_.gcount();
            if (got <= 0) { eof_ = true; continue; }
            buffer_.append(chunk_.data(), static_cast<std::size_t>(got));
            if (!bomStripped_ && readPos_ == 0) {  // strip a UTF-8 BOM once
                if (buffer_.size() >= 3 &&
                    static_cast<unsigned char>(buffer_[0]) == 0xEF &&
                    static_cast<unsigned char>(buffer_[1]) == 0xBB &&
                    static_cast<unsigned char>(buffer_[2]) == 0xBF) {
                    buffer_.erase(0, 3);
                }
                bomStripped_ = true;
            }
        }
    }

    // Empty when next() returned false because of a clean end of input;
    // non-empty describes a parse failure.
    const std::string& error() const { return error_; }

private:
    // A record that parses to a single empty field (trailing newlines etc).
    static bool blankRecord(const std::vector<std::string>& fields) {
        return fields.size() == 1 && fields[0].empty();
    }

    static const std::size_t kChunk = 1u << 20;          // 1 MiB
    static const std::size_t kMaxRecord = 64u << 20;     // safety cap

    std::ifstream& src_;
    std::string buffer_;
    std::size_t readPos_ = 0;   // parse position inside buffer_ (no per-record erase)
    std::vector<char> chunk_;   // allocated once, reused every refill
    bool eof_ = false;
    bool bomStripped_ = false;
    std::string error_;
};

// For each original column, gives its position among the kept columns (-1 if
// dropped). This is so reordered columns still work.
inline std::vector<int> slotMapFromMask(const std::vector<bool>& keepMask) {
    std::vector<int> slot(keepMask.size(), -1);
    int next = 0;
    for (std::size_t i = 0; i < keepMask.size(); ++i) {
        if (keepMask[i]) slot[i] = next++;
    }
    return slot;
}

// SECTION 4 (data): canonical feature list

// Fixed order used for EVERY vector: coordinate i always means FEATURE[i],
// regardless of how the CSV happened to order its columns (the loader maps
// columns by header name, so column reordering cannot silently swap axes).
inline const std::vector<std::string>& featureNames() {
    static const std::vector<std::string> names = {
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo"
    };
    return names;
}

// Per-feature [min, max] observed over the usable songs. A degenerate range
// (min == max) carries no information and normalises every value to 0.0.
struct Normalization {
    std::vector<double> minimum;
    std::vector<double> maximum;
};

// Loader, matcher and recommender

// Aggregate counters reported after loading; nothing is skipped silently.
struct LoadStats {
    std::size_t libraryRows = 0;          // valid-width data rows (excluding header)
    std::size_t libraryNoId = 0;          // rows with neither spotify_id nor catalog_track_id
    std::size_t libraryDuplicates = 0;    // repeated spotify_id rows (first wins)
    std::size_t libraryFallbackKeys = 0;  // records carrying a catalog_track_id
    std::size_t libraryKeyConflicts = 0;  // fallback keys dropped (owned elsewhere/dupe)
    std::size_t libraryMalformed = 0;     // rows with wrong field count
    std::size_t libraryLoaded = 0;        // unique library songs

    std::size_t catalogRows = 0;          // data rows in the catalog
    std::size_t catalogForLibrary = 0;    // catalog rows matching a library id
    std::size_t catalogDuplicates = 0;    // extra feature rows for one id
    std::size_t catalogMalformed = 0;     // rows with wrong field count
    std::size_t rejectedBlank = 0;        // matched but some feature blank
    std::size_t rejectedInvalid = 0;      // matched but some value not a finite number
    std::size_t rejectedImplausible = 0;  // finite but impossible (placeholder rows)
    std::size_t directRejected = 0;       // primary-id rows rejected (song may fall back)
    std::size_t matchedDirect = 0;        // matched via spotify_id
    std::size_t matchedFallback = 0;      // matched only via catalog_track_id
    std::size_t matched = 0;              // matchedDirect + matchedFallback

    std::size_t usable() const { return matched; }
};

// A single ranked result, ready for display.
struct Recommendation {
    SongMeta meta;
    std::size_t vectorIndex = 0;
    double distSq = 0.0;      // squared distance (used for ranking)
    double distance = 0.0;    // Euclidean distance (used for display)
};

// Result of compareSearch(): query timing ONLY -- CSV loading,
// normalization and tree construction are never included in these numbers.
struct PerformanceReport {
    std::size_t points = 0;
    std::size_t dims = 0;
    int k = 0;
    int repeats = 0;
    long long queries = 0;              // how many sample queries were timed
    double treeTimeMicros = 0.0;        // average per query over all samples
    double bruteTimeMicros = 0.0;
    double treeTimeMinMicros = 0.0;     // best/worst single sample: shows noise
    double treeTimeMaxMicros = 0.0;
    double bruteTimeMinMicros = 0.0;
    double bruteTimeMaxMicros = 0.0;
    long long treeNodesVisited = 0;     // average nodes per query
    long long equivalentCount = 0;      // samples where both searches agreed
    bool resultsEquivalent = false;     // all samples agreed
};

class MusicRecommender {
public:
    // Loads the library and catalog. Canonical metadata columns are preferred,
    // with common Spotify-dump aliases accepted; album metadata is optional.
    // The catalog needs track_id plus any of the feature columns (a subset is
    // fine).
    // A song is usable if its id is in the catalog and all selected features
    // are valid numbers. Direct spotify_id matches are done first, then
    // catalog_track_id is tried as a fallback. Duplicates in the library keep
    // the first row.
    bool load(const std::string& libraryCsv, const std::string& catalogCsv,
              std::string& error) {
        error.clear();
        stats_ = LoadStats();
        songs_.clear();
        catalogIds_.clear();
        rawFeatures_.clear();
        std::vector<LibraryRecord>().swap(library_);
        std::vector<std::size_t>().swap(libIndexOf_);
        idOrdinal_ = core::HashTable();
        fallbackOrdinal_ = core::HashTable();
        querySongs_.clear();
        dataset_.clear();
        tree_.clear();
        dims_ = 0;
        norm_ = Normalization();
        idIndex_ = core::HashTable();
        selectedNames_.clear();
        libFeatureCols_.clear();

        // Single-file database: audio features live in the library file's
        // own feature columns (filled by the data merge), so there is no
        // catalog to stream and no fallback resolution to do.
        if (catalogCsv.empty()) {
            if (!readLibrary(libraryCsv, true, error)) return false;
            if (library_.empty()) {
                error = "library contained no songs with a spotify_id";
                return false;
            }
            stats_.matched = stats_.matchedDirect;   // inline rows are their own matches
            return finalizeDataset(error);
        }

        // --- step 1: catalog header (first record), find column positions.
        ColumnMap catalog;
        catalogHeader(catalogCsv, catalog, error);
        if (!error.empty()) return false;
        if (checkRequiredHeader(catalog, "catalog", {"track_id"}, error)) return false;
        if (catalog.names.empty()) {
            error = "catalog has an empty header row";
            return false;
        }
        if (!pickFeatureColumns(catalog, error)) return false;
        if (selectedNames_.empty()) {
            error = "catalog has none of the supported audio-feature columns";
            return false;
        }

        // --- step 2: read the whole library (metadata + id list).
        if (!readLibrary(libraryCsv, false, error)) return false;
        if (library_.empty()) {
            error = "library contained no songs with a spotify_id";
            return false;
        }

        // --- step 2b: register fallback keys now that ALL primary ids are
        // known, so a fallback never claims a catalog id that belongs to
        // another record's spotify_id, independently of CSV row order.
        registerFallbackKeys();

        // --- step 3: stream the catalog twice: direct matches first, then
        // fallback matches for the remaining records.
        if (!scanCatalog(catalogCsv, catalog, error)) return false;

        // --- step 4: assemble vectors, normalization, tree.
        return finalizeDataset(error);
    }

    // Optional extra file of songs to use as query seeds only. They are not
    // added to the search data, so results come from the main library.
    // It needs the same feature columns; rows with bad features are skipped.
    bool loadQuerySongs(const std::string& path, std::string& error) {
        error.clear();
        querySongs_.clear();
        if (dims_ == 0 || selectedNames_.empty()) {
            error = "load the recommendation library before query songs";
            return false;
        }
        std::ifstream file(path, std::ios::binary);
        if (!file) { error = "cannot open query file: " + path; return false; }
        CsvStream csv(file);
        std::vector<std::string> fields;
        if (!csv.next(fields)) {
            error = csv.error().empty() ? "query file has no header row"
                                        : csv.error();
            return false;
        }
        ColumnMap q;
        q.names = fields;
        int idCol = -1, nameCol = -1, artistCol = -1, albumCol = -1;
        int countryCol = -1;
        if (!resolveColumn(q, "query file", {"spotify_id", "id"}, true,
                           idCol, error) ||
            !resolveColumn(q, "query file", {"track_name", "name"}, true,
                           nameCol, error) ||
            !resolveColumn(q, "query file", {"artist_name", "artists"}, true,
                           artistCol, error) ||
            !resolveColumn(q, "query file", {"album_name", "album"}, false,
                           albumCol, error) ||
            !resolveColumn(q, "query file", {"artist_country"}, false,
                           countryCol, error)) {
            return false;
        }

        std::vector<int> featureCols;
        for (const auto& name : selectedNames_) {
            int n = q.count(name);
            if (n > 1) {
                error = "query file: feature column '" + name + "' appears " +
                        std::to_string(n) + " times (ambiguous)";
                return false;
            }
            int c = q.col(name);
            if (c < 0) {
                error = "query file is missing feature column '" + name + "'";
                return false;
            }
            featureCols.push_back(c);
        }

        const std::size_t headerWidth = fields.size();
        core::HashTable seen(1024);
        while (csv.next(fields)) {
            if (fields.size() != headerWidth) continue;
            std::string id = trim(valueAt(fields, static_cast<std::size_t>(idCol)));
            if (id.empty()) continue;
            int dummy;
            if (seen.find(id, dummy)) continue;   // duplicate seed: first wins

            std::vector<double> values;
            values.reserve(featureCols.size());
            bool good = true;
            for (int c : featureCols) {
                std::string text =
                    trim(valueAt(fields, static_cast<std::size_t>(c)));
                double d;
                if (text.empty() || !parseNumber(text, d)) { good = false; break; }
                values.push_back(d);
            }
            if (!good || implausibleFeatures(selectedNames_, values)) continue;

            QuerySong s;
            s.meta.trackId   = id;
            s.meta.trackName = trim(valueAt(fields, static_cast<std::size_t>(nameCol)));
            s.meta.artist    = displayArtist(
                valueAt(fields, static_cast<std::size_t>(artistCol)));
            s.meta.album     = albumCol >= 0
                ? trim(valueAt(fields, static_cast<std::size_t>(albumCol)))
                : std::string();
            s.meta.country   = countryCol >= 0
                ? trim(valueAt(fields, static_cast<std::size_t>(countryCol)))
                : std::string();
            s.features = std::move(values);
            seen.insert(id, static_cast<int>(querySongs_.size()));
            querySongs_.push_back(std::move(s));
        }
        if (!csv.error().empty()) { error = csv.error(); return false; }
        if (querySongs_.empty()) {
            error = "query file has no rows with a usable feature vector";
            return false;
        }
        return true;
    }

    // ---------------- introspection used by the CLI ------------------
    const std::vector<std::string>& selectedFeatures() const { return selectedNames_; }
    std::size_t dims() const { return dims_; }
    const LoadStats& stats() const { return stats_; }
    const Normalization& normalization() const { return norm_; }
    std::size_t songCount() const { return songs_.size(); }
    bool empty() const { return songs_.empty(); }   // nothing usable at all
    int treeMaxDepth() const { return tree_.maxDepth(); }
    std::size_t treeNodes() const { return tree_.nodeCount(); }
    double treeBuildMillis() const { return treeBuildMillis_; }

    // Metadata for a dataset index (bounds-checked).
    const SongMeta& song(std::size_t i) const {
        if (i >= songs_.size()) throw std::out_of_range("MusicRecommender::song");
        return songs_[i];
    }

    // Metadata lookup (generic hash table from core.h)
    // First dataset index for a track id, or false.
    bool indexForTrackId(const std::string& id, std::size_t& indexOut) const {
        int idx;
        if (!idIndex_.find(id, idx)) return false;
        indexOut = static_cast<std::size_t>(idx);
        return true;
    }

    // Case-insensitive substring search on track name and artist. Asks for one
    // more than limit so the caller can tell if there were more matches.
    std::vector<MatchedSong> searchByName(const std::string& text,
                                          std::size_t limit = 8) const {
        return searchMetasByName(
            songs_, [](const SongMeta& s) { return s; }, text, limit);
    }

    // Query-only seed songs (separate CSV, never part of the tree).
    std::size_t querySongCount() const { return querySongs_.size(); }
    const QuerySong& querySong(std::size_t i) const {
        if (i >= querySongs_.size())
            throw std::out_of_range("MusicRecommender::querySong");
        return querySongs_[i];
    }
    bool indexForQueryTrackId(const std::string& id,
                              std::size_t& indexOut) const {
        for (std::size_t i = 0; i < querySongs_.size(); ++i) {
            if (querySongs_[i].meta.trackId == id) { indexOut = i; return true; }
        }
        return false;
    }
    std::vector<MatchedSong> searchQuerySongsByName(
            const std::string& text, std::size_t limit = 8) const {
        return searchMetasByName(
            querySongs_, [](const QuerySong& s) { return s.meta; }, text, limit);
    }

    // Recommendation interface -- ALWAYS uses the K-d tree.
    // Maps every candidate dataset index back to its song metadata.
    // excludeIndex is the query song itself; pass -1 to exclude nothing.
    std::vector<Recommendation> recommend(std::size_t queryIndex, int k,
                                          int excludeIndex = -1,
                                          long long* nodesVisitedOut = nullptr) const {
        if (k <= 0 || tree_.empty() || queryIndex >= dataset_.size()) return {};
        return searchAndRank(dataset_.point(queryIndex), k, excludeIndex,
                             songs_[queryIndex].country, nodesVisitedOut);
    }

    // Same tree query for a vector that is not in the dataset (a seed
    // song): normalized with the corpus ranges, so distances stay
    // comparable with every corpus vector.
    std::vector<Recommendation> recommendByFeatures(
            const std::vector<double>& rawFeatures, int k,
            int excludeIndex = -1,
            long long* nodesVisitedOut = nullptr,
            const std::string& country = "") const {
        if (k <= 0 || tree_.empty() || rawFeatures.size() != dims_) return {};
        std::vector<double> normalized(dims_);
        for (std::size_t f = 0; f < dims_; ++f)
            normalized[f] = normalizeValue(rawFeatures[f], f);
        return searchAndRank(core::VecN(normalized), k, excludeIndex,
                             country, nodesVisitedOut);
    }

    // Same query via brute force -- comparison mode only, never the
    // default recommendation path.
    std::vector<Recommendation> recommendBruteForce(std::size_t queryIndex, int k,
                                                    int excludeIndex = -1) const {
        std::vector<Recommendation> out;
        if (k <= 0 || dataset_.size() == 0 || queryIndex >= dataset_.size()) return out;
        auto candidates = core::bruteForceKNearest(dataset_, dataset_.point(queryIndex),
                                                   k, excludeIndex);
        out.reserve(candidates.size());
        for (const auto& cand : candidates) out.push_back(toRecommendation(cand));
        return out;
    }

    // Runs several evenly spaced queries through both the tree and brute
    // force, repeating each to smooth out timing noise. If checkEquivalence is
    // on, it compares the sorted distances (so ties in a different order are ok).
    PerformanceReport compareSearch(const std::vector<std::size_t>& queryIndices,
                                    int k, int repeats = 25) const {
        PerformanceReport rep;
        if (repeats < 1) repeats = 1;
        rep.repeats = repeats;
        if (k <= 0 || dataset_.size() == 0 || queryIndices.empty()) return rep;
        rep.points = dataset_.size();
        rep.dims = dims_;
        rep.k = k;

        using clock = std::chrono::steady_clock;
        double treeSum = 0.0, bruteSum = 0.0;
        long long nodesSum = 0;
        bool first = true;

        for (std::size_t q : queryIndices) {
            if (q >= dataset_.size()) continue;
            int excludeIndex = static_cast<int>(q);
            const core::VecN& query = dataset_.point(q);

            (void)tree_.searchKNearest(query, k, excludeIndex);   // warm-up run

            auto t0 = clock::now();
            for (int r = 0; r < repeats; ++r) {
                (void)tree_.searchKNearest(query, k, excludeIndex);
            }
            auto t1 = clock::now();
            double treeUs =
                std::chrono::duration<double, std::micro>(t1 - t0).count() / repeats;

            t0 = clock::now();
            for (int r = 0; r < repeats; ++r) {
                (void)core::bruteForceKNearest(dataset_, query, k, excludeIndex);
            }
            t1 = clock::now();
            double bruteUs =
                std::chrono::duration<double, std::micro>(t1 - t0).count() / repeats;

            nodesSum += tree_.lastNodesVisited();
            treeSum += treeUs;
            bruteSum += bruteUs;
            if (first) {
                rep.treeTimeMinMicros = rep.treeTimeMaxMicros = treeUs;
                rep.bruteTimeMinMicros = rep.bruteTimeMaxMicros = bruteUs;
                first = false;
            } else {
                if (treeUs < rep.treeTimeMinMicros)  rep.treeTimeMinMicros  = treeUs;
                if (treeUs > rep.treeTimeMaxMicros)  rep.treeTimeMaxMicros  = treeUs;
                if (bruteUs < rep.bruteTimeMinMicros) rep.bruteTimeMinMicros = bruteUs;
                if (bruteUs > rep.bruteTimeMaxMicros) rep.bruteTimeMaxMicros = bruteUs;
            }
            ++rep.queries;
            if (checkEquivalence(query, k, excludeIndex)) ++rep.equivalentCount;
        }

        if (rep.queries > 0) {
            rep.treeTimeMicros = treeSum / static_cast<double>(rep.queries);
            rep.bruteTimeMicros = bruteSum / static_cast<double>(rep.queries);
            rep.treeNodesVisited = nodesSum / rep.queries;
        }
        rep.resultsEquivalent =
            rep.queries > 0 && rep.equivalentCount == rep.queries;
        return rep;
    }

    // Sample indices for a benchmark sweep: `count` query points evenly
    // spaced across the whole usable dataset (deterministic, so two runs
    // measure the same queries). count is clamped to the dataset size.
    std::vector<std::size_t> benchmarkSampleQueries(std::size_t count) const {
        std::vector<std::size_t> out;
        if (dataset_.size() == 0 || count == 0) return out;
        if (count > dataset_.size()) count = dataset_.size();
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            out.push_back((dataset_.size() * i + count / 2) / count);
        }
        return out;
    }

private:
    // internal pieces

    struct ColumnMap {
        std::vector<std::string> names;
        int col(const std::string& want) const {
            for (std::size_t i = 0; i < names.size(); ++i) {
                if (names[i] == want) return static_cast<int>(i);
            }
            return -1;
        }
        // How many header columns carry this name (0 = missing, >1 =
        // ambiguous: which of the twins would be the data?).
        int count(const std::string& want) const {
            int n = 0;
            for (const auto& name : names) {
                if (name == want) ++n;
            }
            return n;
        }
    };

    // One row per unique key; features get filled in during the catalog scan.
    // Each record has its spotify_id and, as a fallback, the library row's
    // catalog_track_id. meta.trackId always keeps the original Spotify id.
    struct LibraryRecord {
        SongMeta meta;
        std::string catalogId;          // catalog_track_id claimed in the library CSV
        std::string matchedCatalogId;   // catalog row id the features came from
        bool hasFeatures = false;
        std::vector<double> features;   // raw values, canonical feature order
    };

    static void catalogHeader(const std::string& path, ColumnMap& out,
                              std::string& error) {
        std::ifstream file(path, std::ios::binary);
        if (!file) { error = "cannot open file: " + path; return; }
        CsvStream csv(file);   // streaming: header costs one chunk
        std::vector<std::string> header;
        if (!csv.next(header)) {
            error = csv.error().empty() ? "no header row in " + path : csv.error();
            return;
        }
        out.names = header;
    }

    // Header checks: required names must exist exactly once; optional
    // names may be absent but never duplicated.
    static bool checkRequiredHeader(const ColumnMap& m, const std::string& which,
                                    const std::vector<std::string>& required,
                                    std::string& error) {
        for (const auto& name : required) {
            int n = m.count(name);
            if (n == 0) {
                error = which + ": required column '" + name + "' not found";
                return true;
            }
            if (n > 1) {
                error = which + ": required column '" + name + "' appears " +
                        std::to_string(n) + " times (ambiguous)";
                return true;
            }
        }
        return false;
    }

    static bool checkOptionalHeader(const ColumnMap& m, const std::string& which,
                                    const std::string& name, std::string& error) {
        int n = m.count(name);
        if (n > 1) {
            error = which + ": optional column '" + name + "' appears " +
                    std::to_string(n) + " times (ambiguous)";
            return true;
        }
        return false;
    }

    // Resolve a metadata column by its canonical name or one of its aliases.
    // The canonical name wins when both are present; a duplicated candidate is
    // rejected instead of guessing which twin holds the data.
    static bool resolveColumn(const ColumnMap& m, const std::string& which,
                              const std::vector<std::string>& names,
                              bool required, int& out, std::string& error) {
        for (const auto& name : names) {
            int n = m.count(name);
            if (n > 1) {
                error = which + ": column '" + name + "' appears " +
                        std::to_string(n) + " times (ambiguous)";
                return false;
            }
        }
        for (const auto& name : names) {
            int c = m.col(name);
            if (c >= 0) { out = c; return true; }
        }
        if (required) {
            error = which + ": required column '" + names.front() +
                    "' not found";
            return false;
        }
        out = -1;
        return true;
    }

    // The song-metadata columns shared by the library and query-file loaders:
    // id/title/artist are required (with common Spotify-dump aliases), album
    // and artist_country are optional.
    struct MetaColumns {
        int id = -1;
        int name = -1;
        int artist = -1;
        int album = -1;
        int country = -1;
    };

    static bool resolveMetaColumns(const ColumnMap& m, const std::string& which,
                                   MetaColumns& out, std::string& error) {
        return resolveColumn(m, which, {"spotify_id", "id"}, true,
                             out.id, error) &&
               resolveColumn(m, which, {"track_name", "name"}, true,
                             out.name, error) &&
               resolveColumn(m, which, {"artist_name", "artists"}, true,
                             out.artist, error) &&
               resolveColumn(m, which, {"album_name", "album"}, false,
                             out.album, error) &&
               resolveColumn(m, which, {"artist_country"}, false,
                             out.country, error);
    }

    // Intersect the canonical feature list with the catalog's columns.
    // A feature name that appears more than once is ambiguous: which twin
    // column holds the data? Rejected, not silently de-duplicated.
    bool pickFeatureColumns(const ColumnMap& catalog, std::string& error) {
        selectedNames_.clear();
        catalogFeatureCols_.clear();
        for (const auto& f : featureNames()) {
            int n = catalog.count(f);
            if (n == 0) continue;
            if (n > 1) {
                error = "catalog: feature column '" + f + "' appears " +
                        std::to_string(n) + " times (ambiguous)";
                return false;
            }
            selectedNames_.push_back(f);
            catalogFeatureCols_.push_back(catalog.col(f));
        }
        return true;
    }

    // Registers the catalog_track_id fallbacks once the primary ids are final,
    // so a primary id always beats a fallback. A fallback is dropped if it
    // equals the record's own id or another record's primary id; for duplicate
    // fallbacks the earlier row wins and the rest count as conflicts.
    void registerFallbackKeys() {
        fallbackOrdinal_ = core::HashTable(1024);
        for (std::size_t i = 0; i < library_.size(); ++i) {
            const std::string& catId = library_[i].catalogId;
            if (catId.empty() || catId == library_[i].meta.trackId) continue;
            int ord;
            if (idOrdinal_.contains(catId) || fallbackOrdinal_.find(catId, ord)) {
                ++stats_.libraryKeyConflicts;   // first valid registration wins
                continue;
            }
            fallbackOrdinal_.insert(catId, static_cast<int>(i));
        }
    }

    // Reads library metadata (streamed one record at a time) and builds the
    // PRIMARY id -> record-ordinal registry. Fallback ids are stored on the
    // records but NOT registered here; registerFallbackKeys() does that
    // once every primary id is known.
    bool readLibrary(const std::string& path, bool featuresInline,
                     std::string& error) {
        std::ifstream file(path, std::ios::binary);
        if (!file) { error = "cannot open file: " + path; return false; }
        CsvStream csv(file);
        std::vector<std::string> fields;
        if (!csv.next(fields)) {
            error = csv.error().empty() ? "library has no header row" : csv.error();
            return false;
        }

        ColumnMap lib;
        lib.names = fields;
        int idCol = -1, nameCol = -1, artistCol = -1, albumCol = -1;
        int countryCol = -1;
        if (!resolveColumn(lib, "library", {"spotify_id", "id"}, true,
                           idCol, error) ||
            !resolveColumn(lib, "library", {"track_name", "name"}, true,
                           nameCol, error) ||
            !resolveColumn(lib, "library", {"artist_name", "artists"}, true,
                           artistCol, error) ||
            !resolveColumn(lib, "library", {"album_name", "album"}, false,
                           albumCol, error) ||
            !resolveColumn(lib, "library", {"artist_country"}, false,
                           countryCol, error) ||
            checkOptionalHeader(lib, "library", "catalog_track_id", error)) {
            return false;
        }
        int fallbackCol = lib.col("catalog_track_id");   // optional: fallback lookup key

        // Inline-feature files carry the feature columns themselves. A duplicated
        // feature column is an error; missing ones are fine (that axis is just dropped).
        if (featuresInline) {
            for (const auto& name : featureNames()) {
                int n = lib.count(name);
                if (n > 1) {
                    error = "library: feature column '" + name + "' appears " +
                            std::to_string(n) + " times (ambiguous)";
                    return false;
                }
                int c = lib.col(name);
                if (c >= 0) {
                    libFeatureCols_.push_back(c);
                    selectedNames_.push_back(name);
                }
            }
            if (selectedNames_.empty()) {
                error = "library has none of the supported audio-feature columns";
                return false;
            }
        }
        if (!csv.error().empty()) { error = csv.error(); return false; }

        const std::size_t headerWidth = fields.size();
        idOrdinal_ = core::HashTable(1024);
        while (csv.next(fields)) {
            if (fields.size() != headerWidth) {
                ++stats_.libraryMalformed;
                continue;                       // counted, never treated as data
            }
            ++stats_.libraryRows;
            std::string id = trim(valueAt(fields, static_cast<std::size_t>(idCol)));
            std::string catId;
            if (fallbackCol >= 0) {
                catId = trim(valueAt(fields, static_cast<std::size_t>(fallbackCol)));
            }
            if (id.empty() && catId.empty()) { ++stats_.libraryNoId; continue; }

            int ordinal;
            if (!id.empty() && idOrdinal_.find(id, ordinal)) {
                ++stats_.libraryDuplicates;     // duplicate primary id: first wins
                continue;
            }
            if (!catId.empty() && catId != id) ++stats_.libraryFallbackKeys;

            LibraryRecord rec;
            rec.meta.trackId   = id;
            rec.meta.trackName = trim(valueAt(fields, static_cast<std::size_t>(nameCol)));
            rec.meta.artist    = displayArtist(
                valueAt(fields, static_cast<std::size_t>(artistCol)));
            rec.meta.album     = albumCol >= 0
                ? trim(valueAt(fields, static_cast<std::size_t>(albumCol)))
                : std::string();
            rec.meta.country   = countryCol >= 0
                ? trim(valueAt(fields, static_cast<std::size_t>(countryCol)))
                : std::string();
            rec.catalogId      = catId;         // candidate fallback key only
            if (featuresInline) {
                // features are in the row itself; same checks as a catalog row
                // (blank, not a number, or impossible like tempo 0). A rejected row
                // keeps its metadata but gets no features.
                std::vector<double> values(selectedNames_.size(), 0.0);
                bool good = true;
                for (std::size_t f = 0; f < selectedNames_.size(); ++f) {
                    std::string text =
                        trim(valueAt(fields, static_cast<std::size_t>(
                                 libFeatureCols_[f])));
                    if (text.empty()) {
                        ++stats_.rejectedBlank;
                        good = false;
                        break;
                    }
                    double d;
                    if (!parseNumber(text, d)) {
                        ++stats_.rejectedInvalid;
                        good = false;
                        break;
                    }
                    values[f] = d;
                }
                if (good && implausibleFeatures(selectedNames_, values)) {
                    ++stats_.rejectedImplausible;
                    good = false;
                }
                if (good) {
                    rec.features = std::move(values);
                    rec.hasFeatures = true;
                    // the row IS its own match: count it like a direct hit
                    // so the reported totals stay truthful
                    ++stats_.matchedDirect;
                }
            }
            if (!id.empty()) {
                idOrdinal_.insert(id, static_cast<int>(library_.size()));
            }
            library_.push_back(std::move(rec));
            ++stats_.libraryLoaded;
        }
        if (!csv.error().empty()) { error = csv.error(); return false; }
        if (stats_.libraryRows == 0) {
            error = "library has no data rows";
            return false;
        }
        return true;
    }

    // Reads the catalog twice. Pass 1 matches spotify_id, pass 2 matches
    // catalog_track_id for songs that still have no features, so direct matches
    // never lose to a fallback. Only the needed columns are copied.
    struct CatalogSlots {
        std::vector<bool> keepMask;
        int idSlot = -1;
        std::vector<std::size_t> featSlots;
        std::size_t headerWidth = 0;
    };

    CatalogSlots catalogSlots(const ColumnMap& catalog) const {
        CatalogSlots slots;
        int trackIdCol = catalog.col("track_id");
        slots.keepMask.assign(catalog.names.size(), false);
        slots.keepMask[static_cast<std::size_t>(trackIdCol)] = true;
        for (int c : catalogFeatureCols_) {
            slots.keepMask[static_cast<std::size_t>(c)] = true;
        }
        // The slot map translates each original column position into its
        // field position AFTER masking, so a catalog whose columns appear
        // in any order is mapped correctly (kept fields come out sorted by
        // column position).
        std::vector<int> slotMap = slotMapFromMask(slots.keepMask);
        slots.idSlot = slotMap[static_cast<std::size_t>(trackIdCol)];
        slots.featSlots.resize(catalogFeatureCols_.size());
        for (std::size_t f = 0; f < catalogFeatureCols_.size(); ++f) {
            slots.featSlots[f] = static_cast<std::size_t>(
                slotMap[static_cast<std::size_t>(catalogFeatureCols_[f])]);
        }
        slots.headerWidth = catalog.names.size();
        return slots;
    }

    // Rejects placeholder rows from failed audio analysis (tempo 0 with zero
    // danceability/energy, loudness outside [-35, 2] dB). They parse as valid
    // numbers but would stretch the min-max ranges. Only applies to features
    // that are actually selected. Returns true if any value is implausible.
    static bool implausibleFeatures(const std::vector<std::string>& names,
                                    const std::vector<double>& values) {
        std::size_t n = values.size() < names.size() ? values.size() : names.size();
        for (std::size_t f = 0; f < n; ++f) {
            const std::string& name = names[f];
            double v = values[f];
            if (name == "tempo" && v <= 0.0) return true;
            if (name == "energy" && v == 0.0) return true;
            if (name == "danceability" && v == 0.0) return true;
            if (name == "loudness" && (v < -35.0 || v > 2.0)) return true;
        }
        return false;
    }

    // One catalog row: `keys` selects the id registry (primary or
    // fallback); a record already carrying features is never re-filled,
    // so every library entry contributes at most one vector and metadata,
    // catalog id, dataset index and tree results stay aligned.
    void applyCatalogRow(const std::vector<std::string>& kept,
                         const CatalogSlots& slots, const core::HashTable& keys,
                         bool direct) {
        // File-level metrics (rows read, malformed rows) are counted in the
        // direct pass only; pass 2 must not double-count them.
        if (direct) ++stats_.catalogRows;     // pass 2 must not double count
        if (slots.idSlot < 0 || kept.size() <= static_cast<std::size_t>(slots.idSlot)) {
            if (direct) ++stats_.catalogMalformed;   // too few fields: never data
            return;
        }
        std::string id = trim(kept[static_cast<std::size_t>(slots.idSlot)]);
        int ordinal;
        if (id.empty() || !keys.find(id, ordinal)) return;   // not our id
        LibraryRecord& rec = library_[static_cast<std::size_t>(ordinal)];
        if (rec.hasFeatures) { ++stats_.catalogDuplicates; return; }
        ++stats_.catalogForLibrary;

        // every selected feature must parse cleanly
        std::vector<double> values(slots.featSlots.size(), 0.0);
        for (std::size_t f = 0; f < slots.featSlots.size(); ++f) {
            std::string text = (kept.size() > slots.featSlots[f])
                ? trim(kept[slots.featSlots[f]]) : std::string();
            double d;
            if (text.empty()) {
                ++stats_.rejectedBlank;
                if (direct) ++stats_.directRejected;
                return;
            }
            if (!parseNumber(text, d)) {
                ++stats_.rejectedInvalid;
                if (direct) ++stats_.directRejected;
                return;
            }
            values[f] = d;
        }

        // numbers were finite, but the row can still be a placeholder
        // (see implausibleFeatures): reject like blank/invalid rows.
        if (implausibleFeatures(selectedNames_, values)) {
            ++stats_.rejectedImplausible;
            if (direct) ++stats_.directRejected;
            return;
        }

        rec.features = std::move(values);
        rec.hasFeatures = true;
        // The matched catalog row IS the identifier this record's vector
        // came from: keep them aligned for display and dataset mapping.
        rec.matchedCatalogId = id;
        if (direct) ++stats_.matchedDirect; else ++stats_.matchedFallback;
    }

    bool scanCatalog(const std::string& path, const ColumnMap& catalog,
                     std::string& error) {
        CatalogSlots slots = catalogSlots(catalog);

        auto streamWholeCatalog = [&](const core::HashTable& keys, bool direct,
                           std::string& passError) -> bool {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                passError = "cannot open file: " + path;
                return false;
            }
            CsvStream csv(file);
            std::vector<std::string> fields, kept;
            kept.reserve(slots.keepMask.size());
            // Eat the header row again (each pass opens the file fresh);
            // it was already validated by the header reader, and counting
            // it would add a phantom row to catalogRows.
            std::vector<std::string> header;
            if (!csv.next(header)) {
                passError = csv.error().empty()
                    ? "no header row in " + path : csv.error();
                return false;
            }
            while (csv.next(fields)) {
                if (fields.size() != slots.headerWidth) {
                    if (direct) ++stats_.catalogMalformed;
                    continue;
                }
                kept.clear();
                for (std::size_t c = 0; c < slots.keepMask.size(); ++c) {
                    if (slots.keepMask[c]) kept.push_back(fields[c]);
                }
                applyCatalogRow(kept, slots, keys, direct);
            }
            passError = csv.error();
            return passError.empty();
        };

        // pass 1 assigns direct matches first so a fallback NEVER takes
        // features a primary id could have claimed; pass 2 sweeps the
        // leftovers through the fallback table.
        if (!streamWholeCatalog(idOrdinal_, true, error)) return false;
        if (!streamWholeCatalog(fallbackOrdinal_, false, error)) return false;
        stats_.matched = stats_.matchedDirect + stats_.matchedFallback;
        return true;
    }

    // normalization + SECTION 5: vector construction
    bool finalizeDataset(std::string& error) {
        (void)error;
        // Songs with a usable catalog row, in original library order.
        songs_.clear();
        catalogIds_.clear();
        rawFeatures_.clear();
        libIndexOf_.clear();
        for (std::size_t i = 0; i < library_.size(); ++i) {
            const auto& rec = library_[i];
            if (!rec.hasFeatures) continue;
            songs_.push_back(rec.meta);
            catalogIds_.push_back(rec.matchedCatalogId);   // real vector source
            rawFeatures_.push_back(rec.features);
            libIndexOf_.push_back(i);
        }
        if (songs_.empty()) {
            error = "no library song had usable features in the catalog "
                    "(see the loading report above for the accounting)";
            return false;
        }
        dims_ = selectedNames_.size();

        // min-max per feature over the whole usable corpus (one scale for
        // every dimension so distances are comparable across features)
        norm_.minimum.assign(dims_, 0.0);
        norm_.maximum.assign(dims_, 0.0);
        for (std::size_t f = 0; f < dims_; ++f) {
            double lo = rawFeatures_[0][f], hi = lo;
            for (const auto& row : rawFeatures_) {
                if (row[f] < lo) lo = row[f];
                if (row[f] > hi) hi = row[f];
            }
            norm_.minimum[f] = lo;
            norm_.maximum[f] = hi;
        }

        // build the normalized VecN dataset; every coordinate must come out
        // finite (a non-finite source value or degenerate range would poison
        // the tree), and constant features map to 0.0 without division.
        for (const auto& row : rawFeatures_) {
            std::vector<double> coords(dims_);
            for (std::size_t f = 0; f < dims_; ++f) {
                double nv = normalizeValue(row[f], f);
                if (!std::isfinite(nv)) {
                    error = "non-finite coordinate produced while normalizing "
                            "feature '" + selectedNames_[f] + "'";
                    return false;
                }
                coords[f] = nv;
            }
            dataset_.add(core::VecN(coords));
        }

        // id -> dataset index. Two passes so a primary spotify_id always wins:
        // pass 1 adds every song's own id (first one wins), pass 2 adds claimed
        // catalog_track_ids only if no one already owns that key.
        idIndex_ = core::HashTable(std::max<std::size_t>(1024, dataset_.size() / 8));
        // pass 1: every song's own spotify_id (whoever owns it, wins)
        for (std::size_t i = 0; i < dataset_.size(); ++i) {
            if (!songs_[i].trackId.empty() && !idIndex_.contains(songs_[i].trackId)) {
                idIndex_.insert(songs_[i].trackId, static_cast<int>(i));
            }
        }
        // pass 2: claimed catalog_track_ids, but ONLY for keys nobody
        // already owns -- this order is what keeps primaries ahead.
        for (std::size_t i = 0; i < dataset_.size(); ++i) {
            const std::string& claimed = library_[libIndexOf_[i]].catalogId;
            if (!claimed.empty() && !idIndex_.contains(claimed)) {
                idIndex_.insert(claimed, static_cast<int>(i));
            }
        }

        // build the tree; construction time is kept separately from query
        // times so the benchmark only compares searches
        using clock = std::chrono::steady_clock;   // steady = not wall clock
        auto buildStart = clock::now();
        bool ok = tree_.build(dataset_);
        auto buildDone = clock::now();
        treeBuildMillis_ =
            std::chrono::duration<double, std::milli>(buildDone - buildStart)
                .count();

        // The tree and normalized songs are the only structures needed after
        // loading. Release the loader-side copies for large corpora.
        std::vector<LibraryRecord>().swap(library_);
        std::vector<std::string>().swap(catalogIds_);
        std::vector<std::vector<double>>().swap(rawFeatures_);
        std::vector<std::size_t>().swap(libIndexOf_);
        idOrdinal_ = core::HashTable();
        fallbackOrdinal_ = core::HashTable();
        return ok;
    }

    // Min-max normalize one raw value for feature f; degenerate range -> 0.
    double normalizeValue(double raw, std::size_t f) const {
        double range = norm_.maximum[f] - norm_.minimum[f];
        if (range <= 0.0) return 0.0;
        return (raw - norm_.minimum[f]) / range;
    }

    static int countryCandidateCount(int k) {
        long long expanded = std::max<long long>(k + 100, k * 20LL);
        return static_cast<int>(std::min<long long>(expanded, 1000));
    }

    // Splits a ';'-separated ISO code list ("IN;PK") into trimmed codes.
    static std::vector<std::string> splitCountryCodes(const std::string& list) {
        std::vector<std::string> codes;
        std::size_t start = 0;
        while (start < list.size()) {
            std::size_t end = list.find(';', start);
            if (end == std::string::npos) end = list.size();
            codes.push_back(trim(list.substr(start, end - start)));
            start = end + 1;
        }
        return codes;
    }

    // True when the two ';'-separated code lists share at least one code.
    static bool sharesCountry(const std::string& left,
                              const std::string& right) {
        if (left.empty() || right.empty()) return false;
        for (const std::string& code : splitCountryCodes(left)) {
            for (const std::string& other : splitCountryCodes(right)) {
                if (code == other) return true;
            }
        }
        return false;
    }

    // Re-ranks an expanded candidate pool: audio distance is always the
    // primary signal, with a fixed penalty for a country mismatch (or a
    // milder one when the candidate has no country at all). The K-d tree
    // never sees country -- this is display-order only, so the reported
    // distance stays pure audio similarity.
    static constexpr double kCountryMismatchPenalty = 0.35;
    static constexpr double kUnknownCountryPenalty = 0.175;   // half of the above

    std::vector<Recommendation> rankRecommendations(
            const std::vector<core::Candidate>& candidates, int k,
            const std::string& queryCountry) const {
        struct Ranked {
            Recommendation recommendation;
            double score = 0.0;
        };
        std::vector<Ranked> ranked;
        ranked.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            Recommendation recommendation = toRecommendation(candidate);
            double countryPenalty = 0.0;
            if (!queryCountry.empty()) {
                if (recommendation.meta.country.empty())
                    countryPenalty = kUnknownCountryPenalty;
                else if (!sharesCountry(queryCountry, recommendation.meta.country))
                    countryPenalty = kCountryMismatchPenalty;
            }
            ranked.push_back({recommendation,
                              recommendation.distance + countryPenalty});
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const Ranked& a, const Ranked& b) {
                      if (a.score != b.score) return a.score < b.score;
                      return a.recommendation.distSq < b.recommendation.distSq;
                  });
        if (ranked.size() > static_cast<std::size_t>(k)) ranked.resize(k);
        std::vector<Recommendation> out;
        out.reserve(ranked.size());
        for (auto& item : ranked) out.push_back(std::move(item.recommendation));
        return out;
    }

    Recommendation toRecommendation(const core::Candidate& cand) const {
        Recommendation r;
        std::size_t idx = static_cast<std::size_t>(std::max(cand.index, 0));
        if (idx < songs_.size()) {
            r.meta = songs_[idx];
            r.vectorIndex = idx;
        }
        r.distSq = cand.distSq;
        r.distance = std::sqrt(std::max(0.0, cand.distSq));
        return r;
    }

    // Shared tree-search path for every recommendation entry point. When the
    // query carries a country, asks the tree for an oversized candidate pool
    // (so same-country tracks have something to win with) and lets
    // rankRecommendations() pick the final k; the pool size is the only
    // country influence, never a K-d tree axis.
    std::vector<Recommendation> searchAndRank(
            const core::VecN& query, int k, int excludeIndex,
            const std::string& queryCountry,
            long long* nodesVisitedOut) const {
        const int candidateK =
            queryCountry.empty() ? k : countryCandidateCount(k);
        auto candidates =
            tree_.searchKNearest(query, candidateK, excludeIndex);
        if (nodesVisitedOut) *nodesVisitedOut = tree_.lastNodesVisited();
        return rankRecommendations(candidates, k, queryCountry);
    }

    // Case-insensitive substring search over any song list. metaOf projects
    // each element onto its SongMeta (corpus rows and seed songs share this).
    // Asks for one extra match so callers can tell "there were more".
    template <typename SongList, typename MetaOf>
    static std::vector<MatchedSong> searchMetasByName(
            const SongList& songs, MetaOf metaOf, const std::string& text,
            std::size_t limit) {
        std::string key = toLower(trim(text));
        std::vector<MatchedSong> out;
        if (key.empty()) return out;
        const std::size_t cap = limit + 1;
        for (std::size_t i = 0; i < songs.size() && out.size() < cap; ++i) {
            const SongMeta& meta = metaOf(songs[i]);
            std::string hay = toLower(meta.trackName + " " + meta.artist);
            if (hay.find(key) != std::string::npos) {
                out.push_back({meta, i});
            }
        }
        return out;
    }

    // Equivalence allowing ties: the two searches may keep different members
    // of a tie group in different orders, so we compare the SORTED (already
    // ascending) sequences of squared distances element by element.
    bool checkEquivalence(const core::VecN& query, int k, int excludeIndex) const {
        auto a = tree_.searchKNearest(query, k, excludeIndex);
        auto b = core::bruteForceKNearest(dataset_, query, k, excludeIndex);
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (std::fabs(a[i].distSq - b[i].distSq) > 1e-9) return false;
        }
        return true;
    }

    // ---- small helpers ----
    static std::string valueAt(const std::vector<std::string>& row, std::size_t i) {
        return i < row.size() ? row[i] : std::string();
    }
    static std::string toLower(std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }
    static std::string trim(const std::string& s) {
        std::size_t a = 0, b = s.size();
        while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        return s.substr(a, b - a);
    }

    // Some datasets store artists as a Python-style list literal, e.g.
    // "['Artist A', 'Artist B']". Turn that into "Artist A, Artist B" for
    // display and name search; plain strings pass through unchanged.
    static std::string displayArtist(const std::string& raw) {
        std::string s = trim(raw);
        if (s.size() < 2 || s.front() != '[' || s.back() != ']') return s;
        std::string out;
        std::size_t i = 1;
        const std::size_t end = s.size() - 1;
        while (i < end) {
            char c = s[i];
            if (c != '\'' && c != '"') { ++i; continue; }
            char quote = c;
            ++i;
            std::string part;
            while (i < end) {
                char ch = s[i++];
                if (ch == '\\' && i < end) { part += s[i++]; continue; }
                if (ch == quote) break;
                part += ch;
            }
            if (!out.empty()) out += ", ";
            out += part;
        }
        return out;
    }
    // Strict number parse: the whole token must be a finite number, so "12abc",
    // NaN and inf fail instead of silently turning into 0.
    static bool parseNumber(const std::string& s, double& out) {
        try {
            std::size_t used = 0;
            double v = std::stod(s, &used);
            while (used < s.size() &&
                   std::isspace(static_cast<unsigned char>(s[used]))) ++used;
            if (used != s.size()) return false;    // trailing junk
            if (!std::isfinite(v)) return false;   // NaN / +inf / -inf
            out = v;
            return true;
        } catch (...) {
            return false;
        }
    }

    // ---- data members ----
    std::vector<LibraryRecord> library_;   // one per unique library id
    std::vector<SongMeta> songs_;          // index i == dataset index i
    std::vector<std::string> catalogIds_;  // parallel to songs_: fallback keys
    std::vector<std::vector<double>> rawFeatures_; // parallel to songs_
    core::KDataset dataset_;               // declared before tree_: tree_ holds
    core::KDTree tree_;                    // a pointer into dataset_, so it must
                                           // be destroyed first (guaranteed by
                                           // reverse declaration order)
    Normalization norm_;
    LoadStats stats_;
    std::vector<std::string> selectedNames_;   // usable feature names, order = axes
    // Inline-feature mode: the library columns each selected feature reads
    // from (same order as selectedNames_); empty in two-file catalog mode.
    std::vector<int> libFeatureCols_;
    std::vector<int> catalogFeatureCols_;      // their columns in the catalog
    std::vector<QuerySong> querySongs_;        // optional --seeds file (query only)
    core::HashTable idOrdinal_;                // library pass: primary id -> record ordinal
    core::HashTable fallbackOrdinal_;          // fallback id -> record ordinal (built after)
    core::HashTable idIndex_;                  // lookup: id -> dataset index
    std::size_t dims_ = 0;
    std::vector<std::size_t> libIndexOf_;  // dataset index -> library_ record
    double treeBuildMillis_ = 0.0;             // measured in finalizeDataset
};

// Result formatting

// Prints the ranked table. Distance is the plain Euclidean distance across
// the SELECTED, MIN-MAX NORMALIZED features -- smaller means "closer in the
// feature space", which is not the same as "more like my taste" (taste lives
// in the playlists, not in the numbers; see README).
inline void printRecommendationsFor(const SongMeta& q,
                                    const std::vector<Recommendation>& results) {
    std::cout << "\nRecommendations for \"" << q.trackName << "\" -- "
              << q.artist << "\n(rank, distance, title -- artist -- album)\n";
    if (results.empty()) {
        std::cout << "  (no candidates: dataset too small or k <= 0)\n";
        return;
    }
    int rank = 1;
    for (const auto& r : results) {
        std::cout << "  " << rank++ << ". d=" << r.distance
                  << "  \"" << r.meta.trackName << "\" -- "
                  << r.meta.artist << " -- "
                  << (r.meta.album.empty() ? "(unknown album)" : r.meta.album)
                  << "\n";
    }
    std::cout << "(smaller distance = more similar across the selected "
                 "normalized audio features only)\n";
}

inline void printRecommendations(const MusicRecommender& rec,
                                 std::size_t queryIndex,
                                 const std::vector<Recommendation>& results) {
    printRecommendationsFor(rec.song(queryIndex), results);
}

#endif // MUSIC_H
