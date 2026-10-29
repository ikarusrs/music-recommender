// main.cpp -- command-line menu for the recommender.
// Loads the data, builds the tree, then loops: recommend by track id, search
// by name, compare K-d tree vs brute force, dataset info, quit. Normal
// recommendations always use the tree; brute force only runs in the
// comparison, which times the search only (not loading or building).

#include "core.h"
#include "music.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// input helpers

// prints the prompt and reads a line; sets eofOut on end of input
std::string promptLine(const std::string& prompt, bool& eofOut) {
    std::cout << prompt << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) {
        std::cin.clear();
        eofOut = true;
        return std::string();
    }
    eofOut = false;
    return line;
}

int promptInt(const std::string& prompt, int fallback, bool& eofOut) {
    std::string line = promptLine(prompt, eofOut);
    try {
        return std::stoi(line);
    } catch (const std::out_of_range&) {
        // A number too big for int is still valid input: report it far above
        // any dataset size so clampK clamps it and prints its message.
        return std::numeric_limits<int>::max();
    } catch (...) {
        return fallback;   // empty, or not a number: take the default
    }
}

// Validates a requested neighbour count against the loaded dataset and
// clamps it to the dataset size, so a junk number like 99999999 cannot make
// the bounded heap reserve that many slots.
int clampK(MusicRecommender& rec, int k) {
    if (k <= 0) return 0;
    int size = static_cast<int>(rec.songCount());
    if (k > size) {
        std::cout << "K is larger than the dataset; clamped to " << size << "\n";
        k = size;
    }
    return k;
}

// report helpers

void printLoadReport(const MusicRecommender& rec, bool verbose) {
    const LoadStats& s = rec.stats();
    std::cout << "\nData loading report\n";

    // One summary line by default -- the full accounting only matters when
    // something looks suspicious, so it lives behind --debug.
    if (!verbose && !rec.empty()) {
        long rejected = static_cast<long>(s.rejectedBlank) +
                        static_cast<long>(s.rejectedInvalid) +
                        static_cast<long>(s.rejectedImplausible);
        std::cout << "  "
                  << rec.songCount() << " usable songs, "
                  << rec.dims() << " features; K-d tree ("
                  << rec.treeNodes() << " nodes, depth "
                  << rec.treeMaxDepth() << ") built in "
                  << rec.treeBuildMillis() << " ms";
        if (rejected > 0)
            std::cout << "; " << rejected << " rows rejected";
        std::cout << "\n";
        std::cout << "  (--debug shows the full accounting)\n";
        if (rec.songCount() < 12) {
            std::cout << "\nWARNING: fewer than 12 usable songs; recommendations "
                         "will be weak and the K-d tree small.\n";
        }
        return;
    }

    std::cout << "  library rows read          : " << s.libraryRows << "\n";
    std::cout << "  library rows w/o track id  : " << s.libraryNoId << "\n";
    std::cout << "  duplicate library ids      : " << s.libraryDuplicates
              << " (first occurrence kept)\n";
    std::cout << "  fallback catalog_track_id  : " << s.libraryFallbackKeys
              << " (used when spotify_id is missing from the catalog)\n";
    std::cout << "  conflicting fallback keys  : " << s.libraryKeyConflicts
              << " (two library rows aimed at one catalog row; first wins)\n";
    std::cout << "  unique library songs       : " << s.libraryLoaded << "\n";
    std::cout << "  malformed library rows     : " << s.libraryMalformed
              << " (wrong field count; skipped)\n";
    std::cout << "  feature catalog rows       : " << s.catalogRows << "\n";
    std::cout << "  malformed catalog rows     : " << s.catalogMalformed
              << " (wrong field count; skipped)\n";
    std::cout << "  catalog rows for your ids  : " << s.catalogForLibrary << "\n";
    std::cout << "  duplicate feature rows     : " << s.catalogDuplicates << "\n";
    std::cout << "  rejected (blank features)  : " << s.rejectedBlank << "\n";
    std::cout << "  rejected (invalid numbers) : " << s.rejectedInvalid << "\n";
    std::cout << "  rejected (implausible)     : " << s.rejectedImplausible
              << " (placeholder rows, e.g. tempo 0)\n";
    std::cout << "  rejected direct-id rows    : " << s.directRejected
              << " (song still usable via its fallback id)\n";
    std::cout << "  matched via spotify id     : " << s.matchedDirect << "\n";
    std::cout << "  matched via fallback id    : " << s.matchedFallback << "\n";
    std::cout << "  matched usable songs       : " << s.matched << "\n";
    std::cout << "  usable features (" << rec.dims() << "): ";
    for (const auto& f : rec.selectedFeatures()) std::cout << f << " ";
    std::cout << "\n";
    std::cout << "  K-d tree nodes             : " << rec.treeNodes() << "\n";
    std::cout << "  K-d tree depth             : " << rec.treeMaxDepth() << "\n";
    std::cout << "  K-d tree construction      : " << rec.treeBuildMillis()
              << " ms (build only; separate from query timings)\n";

    // Show normalization range so the numbers on screen are accountable.
    const Normalization& n = rec.normalization();
    std::cout << "  normalization [min..max]   :\n";
    for (std::size_t f = 0; f < rec.dims(); ++f) {
        std::cout << "    " << rec.selectedFeatures()[f] << ": "
                  << n.minimum[f] << " .. " << n.maximum[f]
                  << (n.maximum[f] > n.minimum[f] ? "" : "  (constant; uses 0.0)")
                  << "\n";
    }

    if (rec.empty()) {
        std::cout << "\nNOT ENOUGH DATA: no library song had usable features in "
                     "the catalog, so no tree was built.\n";
    } else if (rec.songCount() < 12) {
        std::cout << "\nWARNING: fewer than 12 usable songs; recommendations "
                     "will be weak and the K-d tree small.\n";
    }
}

void printComparison(const PerformanceReport& rep) {
    std::cout << "\nQuery performance comparison (benchmark sweep)\n";
    std::cout << "  sample queries       : " << rep.queries
              << " (evenly spaced across the dataset)\n";
    std::cout << "  repeats per query    : " << rep.repeats << "\n";
    std::cout << "  usable points        : " << rep.points << "\n";
    std::cout << "  dimensions           : " << rep.dims << "\n";
    std::cout << "  requested K          : " << rep.k << "\n";
    std::cout << "  K-d tree query time  : " << rep.treeTimeMicros
              << " us per query (avg); best " << rep.treeTimeMinMicros
              << ", worst " << rep.treeTimeMaxMicros << "\n";
    std::cout << "  K-d tree nodes visited per query : " << rep.treeNodesVisited
              << " of " << rep.points << " total\n";
    std::cout << "  brute force time     : " << rep.bruteTimeMicros
              << " us per query (avg); best " << rep.bruteTimeMinMicros
              << ", worst " << rep.bruteTimeMaxMicros << "\n";
    std::cout << "  equivalent results (ties ignored): "
              << rep.equivalentCount << " of " << rep.queries << "\n";
    if (rep.treeTimeMicros > 0 && rep.bruteTimeMicros > 0) {
        std::cout << "  measured speedup     : "
                  << (rep.bruteTimeMicros / rep.treeTimeMicros) << "x "
                  "(machine-, dataset-, dimension- and query-dependent)\n";
    }
}

} // namespace

// menu actions

namespace {

struct QueryPick {
    bool fromSeeds = false;   // true = query-only --seeds row
    std::size_t index = 0;    // corpus index or seed index
};

int resolveQuerySongById(MusicRecommender& rec, QueryPick& pick, bool& eof) {
    std::string id = promptLine("Track ID: ", eof);
    if (eof) return -2;
    if (id.empty()) {
        std::cout << "Cancelled.\n";
        return -1;
    }
    std::size_t index;
    if (rec.indexForQueryTrackId(id, index)) {
        pick = {true, index};
    } else if (rec.indexForTrackId(id, index)) {
        pick = {false, index};
    } else {
        std::cout << "Track id \"" << id
                  << "\" is not in the recommendation dataset.\n";
        return -1;
    }
    const SongMeta& m = pick.fromSeeds ? rec.querySong(pick.index).meta
                                       : rec.song(pick.index);
    std::cout << "Found: \"" << m.trackName << "\" -- " << m.artist << "\n";
    return 0;
}

int resolveQuerySongByName(MusicRecommender& rec, QueryPick& pick, bool& eof) {
    std::string text = promptLine("Search text (title or artist): ", eof);
    if (eof) return -2;
    if (text.empty()) {
        std::cout << "Cancelled.\n";
        return -1;
    }
    // Both databases are searched: --seeds rows first (they are the songs
    // the user explicitly brought to query), then the recommendation corpus.
    struct Hit { QueryPick pick; SongMeta meta; };
    std::vector<Hit> hits;
    for (const auto& h : rec.searchQuerySongsByName(text))
        hits.push_back({{true, h.vectorIndex}, h.meta});
    for (const auto& h : rec.searchByName(text))
        hits.push_back({{false, h.vectorIndex}, h.meta});
    if (hits.empty()) {
        std::cout << "No usable song matches \"" << text << "\".\n";
        return -1;
    }
    // Each search fetches limit+1 candidates; anything beyond the shown
    // limit means more matches exist.
    const std::size_t limit = 8;
    const std::size_t shown = std::min(hits.size(), limit);
    std::cout << "Matches:\n";
    for (std::size_t i = 0; i < shown; ++i) {
        const auto& h = hits[i];
        std::cout << "  " << (i + 1) << ". " << h.meta.trackName << " -- "
                  << h.meta.artist
                  << (h.meta.album.empty() ? "" : " (" + h.meta.album + ")")
                  << "\n";
    }
    if (hits.size() > limit) {
        std::cout << "(more matches exist; refine your search)\n";
    }
    std::string pickText = promptLine("Pick 1-" + std::to_string(shown) +
                                      " (empty = cancel): ", eof);
    if (eof || pickText.empty()) return -1;
    int choice;
    try { choice = std::stoi(pickText); } catch (...) {
        std::cout << "Invalid choice; cancelled.\n";
        return -1;
    }
    if (choice < 1 || static_cast<std::size_t>(choice) > shown) {
        std::cout << "Invalid choice; cancelled.\n";
        return -1;
    }
    pick = hits[static_cast<std::size_t>(choice - 1)].pick;
    return 0;
}

int resolveQuerySeedFromList(MusicRecommender& rec, QueryPick& pick,
                                 bool& eof) {
    if (rec.querySongCount() == 0) {
        std::cout << "No seed songs are loaded.\n";
        return -1;
    }
    std::cout << "\nLata Mangeshkar seed songs:\n";
    for (std::size_t i = 0; i < rec.querySongCount(); ++i) {
        const SongMeta& m = rec.querySong(i).meta;
        std::cout << "  " << (i + 1) << ". " << m.trackName << " -- "
                  << m.artist
                  << (m.album.empty() ? "" : " (" + m.album + ")")
                  << "\n";
    }
    std::string pickText = promptLine(
        "Pick 1-" + std::to_string(rec.querySongCount()) +
        " (empty = cancel): ", eof);
    if (eof || pickText.empty()) {
        std::cout << "Cancelled.\n";
        return -1;
    }
    int choice;
    try { choice = std::stoi(pickText); } catch (...) {
        std::cout << "Invalid choice; cancelled.\n";
        return -1;
    }
    if (choice < 1 ||
        static_cast<std::size_t>(choice) > rec.querySongCount()) {
        std::cout << "Invalid choice; cancelled.\n";
        return -1;
    }
    pick = {true, static_cast<std::size_t>(choice - 1)};
    return 0;
}

// Runs the tree for either a corpus row or a seed song (normalized with
// the corpus ranges); corpus songs are excluded from their own answer.
std::vector<Recommendation> recommendForPick(const MusicRecommender& rec,
                                             const QueryPick& pick, int k) {
    if (!pick.fromSeeds) {
        return rec.recommend(pick.index, k, static_cast<int>(pick.index));
    }
    const QuerySong& q = rec.querySong(pick.index);
    int exclude = -1;
    std::size_t corpusIndex;
    if (rec.indexForTrackId(q.meta.trackId, corpusIndex))
        exclude = static_cast<int>(corpusIndex);
    return rec.recommendByFeatures(q.features, k, exclude, nullptr,
                                   q.meta.country);
}

void printPickRecommendations(const MusicRecommender& rec,
                              const QueryPick& pick,
                              const std::vector<Recommendation>& recs) {
    if (pick.fromSeeds)
        printRecommendationsFor(rec.querySong(pick.index).meta, recs);
    else
        printRecommendations(rec, pick.index, recs);
}

void printDatasetInfo(const MusicRecommender& rec, bool debug) {
    std::cout << "\nDataset\n";
    std::cout << "  usable songs      : " << rec.songCount() << "\n";
    std::cout << "  query seeds       : " << rec.querySongCount() << "\n";
    std::cout << "  K-d tree nodes    : " << rec.treeNodes() << "\n";
    std::cout << "  K-d tree depth    : " << rec.treeMaxDepth() << "\n";
    // Feature axes and normalization ranges, for debugging.
    if (!debug) {
        std::cout << "  features          : " << rec.dims()
                  << " (--debug lists them with their normalization ranges)\n";
        return;
    }
    std::cout << "  features          : " << rec.dims() << " (";
    for (std::size_t i = 0; i < rec.dims(); ++i) std::cout << rec.selectedFeatures()[i]
        << (i + 1 < rec.dims() ? ", " : "");
    std::cout << ")\n";
    std::cout << "  normalization     : min-max over the usable corpus "
                 "(degenerate ranges -> 0.0)\n";
    const Normalization& n = rec.normalization();
    for (std::size_t f = 0; f < rec.dims(); ++f) {
        std::cout << "    " << rec.selectedFeatures()[f] << ": "
                  << n.minimum[f] << " .. " << n.maximum[f]
                  << (n.maximum[f] > n.minimum[f] ? "" : "  (constant; uses 0.0)")
                  << "\n";
    }
}

} // namespace

// main

int main(int argc, char** argv) {
    // The generated union of all local sources is the default corpus. The
    // smaller merged library remains available through --library.
    std::string libraryPath = "db/recommendation_pool.csv";
    std::string seedsPath = "db/lata_seed_pool.csv";

    bool debug = false;
    bool explicitSeeds = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            debug = true;
        } else if (arg == "--library" || arg == "--seeds") {
            if (i + 1 >= argc) {
                std::cerr << "usage: ./music-recommender [--library FILE] "
                             "[--seeds FILE] [--debug]\n";
                return 1;
            }
            std::string value = argv[++i];
            if (arg == "--library") {
                libraryPath = value;
            } else {
                seedsPath = value;
                explicitSeeds = true;
            }
        } else if (arg.rfind("--library=", 0) == 0) {
            libraryPath = arg.substr(10);
        } else if (arg.rfind("--seeds=", 0) == 0) {
            seedsPath = arg.substr(8);
            explicitSeeds = true;
        } else {
            std::cerr << "usage: ./music-recommender [--library FILE] "
                         "[--seeds FILE] [--debug]\n";
            return 1;
        }
    }


    std::cout << "Vector music recommender (K-d tree nearest neighbours)\n";
    std::cout << "library : " << libraryPath
              << " (features inline in the rows)\n";

    MusicRecommender rec;
    std::string error;
    bool built = rec.load(libraryPath, "", error);
    // Full accounting when debugging or when the load failed (failure
    // cases need the whole breakdown to be actionable).
    printLoadReport(rec, debug || !built);
    if (!built) {
        std::cerr << "\nLOAD FAILED: " << error << "\n";
        return 1;
    }

    if (rec.empty()) {
        std::cerr << "\nNothing to recommend from. Fix the data files and "
                     "try again.\n";
        return 1;
    }

    {
        std::ifstream seedProbe(seedsPath);
        if (seedProbe.good()) {
            if (!rec.loadQuerySongs(seedsPath, error)) {
                std::cerr << "\nQUERY LOAD FAILED: " << error << "\n";
                return 1;
            }
            std::cout << "queries : " << seedsPath << " ("
                      << rec.querySongCount() << " seed songs)\n";
        } else if (explicitSeeds) {
            std::cerr << "\nQUERY LOAD FAILED: cannot open query file: "
                      << seedsPath << "\n";
            return 1;
        }
    }

    int defaultK = 10;   // configurable per query
    for (;;) {                       // menu loop: runs until Quit/EOF
        std::cout << "\nMenu\n"
                     "  1. Lata Mangeshkar seed songs\n"
                     "  2. Find a song by title/artist, then recommend\n"
                     "  3. Recommend by track ID\n"
                     "  4. Compare K-d tree vs brute force search\n"
                     "  5. Dataset info\n"
                     "  0. Quit\n";
        bool eof = false;
        std::string choice = promptLine("> ", eof);
        // EOF at the menu is a polite quit too (piping `q` in works the same)
        if (eof || (!choice.empty() && (choice[0] == 'q' || choice[0] == 'Q'))) break;
        int status = 0;
        QueryPick pick;
        if (choice == "1") {
            status = resolveQuerySeedFromList(rec, pick, eof);
        } else if (choice == "2") {
            status = resolveQuerySongByName(rec, pick, eof);
        } else if (choice == "3") {
            status = resolveQuerySongById(rec, pick, eof);
        } else if (choice == "4") {
            // Multi-query benchmark: no need to pick one track -- the sweep
            // samples evenly spaced queries across the whole dataset (no
            // cherry-picking the fast cases).
            int k = promptInt("How many candidates to keep? [10]: ", defaultK, eof);
            k = clampK(rec, k);
            if (k <= 0) { std::cout << "K must be >= 1\n"; continue; }
            int defaultRepeats = rec.songCount() > 100000 ? 3 : 25;
            int repeats = promptInt(
                "Repetitions per query [" + std::to_string(defaultRepeats) +
                    "]: ", defaultRepeats, eof);
            if (repeats < 1) repeats = 1;
            auto samples = rec.benchmarkSampleQueries(25);
            std::cout << "Benchmarking " << samples.size()
                      << " evenly spaced queries...\n";
            printComparison(rec.compareSearch(samples, k, repeats));
            continue;
        } else if (choice == "5") {
            printDatasetInfo(rec, debug);  // --debug adds features/ranges
            continue;
        } else if (choice == "0") {
            break;
        } else {
            std::cout << "Unknown option (0-5).\n";
            continue;
        }
        if (status == -2) break;
        if (status < 0) continue;
        int k = promptInt("How many recommendations? [10]: ", defaultK, eof);
        k = clampK(rec, k);
        if (k <= 0) { std::cout << "K must be >= 1\n"; continue; }
        auto recs = recommendForPick(rec, pick, k);
        printPickRecommendations(rec, pick, recs);
    }
    std::cout << "Bye.\n";
    return 0;
}
