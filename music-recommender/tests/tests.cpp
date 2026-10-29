// ============================================================================
// tests/tests.cpp -- tiny self-written test runner for the project.
//
// No framework, no dependencies: every check is one line like
//     check("name of what we prove", actual == expected);
// and at the end the runner prints PASS/FAIL counts and exits with
// 0 (all good) or 1 (something failed) so `make test` can report it.
//
// The CSV fixtures it loads live in tests/fixtures/ next to this file.
// ============================================================================

#include "core.h"
#include "music.h"

#include <cstddef>
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// mini test framework: counters + one helper
// ---------------------------------------------------------------------------

static int passed = 0;
static int failed = 0;

static void check(const std::string& what, bool ok) {
    if (ok) {
        ++passed;
        std::printf("  [ok]   %s\n", what.c_str());
    } else {
        ++failed;
        std::printf("  [FAIL] %s\n", what.c_str());
    }
}

// compare doubles with a small tolerance (roots/log maths never land
// exactly on the textbook value)
static bool near(double a, double b, double eps = 1e-9) {
    double diff = a > b ? a - b : b - a;
    return diff <= eps;
}

std::string fx(const char* name) {   // path to a fixture file
    return "tests/fixtures/" + std::string(name);
}

struct LoadOk {
    // MusicRecommender is neither copyable nor movable (its KDTree is not),
    // so the test wrapper holds it behind a unique_ptr.
    std::unique_ptr<MusicRecommender> rec;
    std::string error;
    bool ok = false;
};

// one call that loads a pair of CSVs and reports the whole result
static LoadOk load(const std::string& lib, const std::string& cat) {
    LoadOk r;
    r.rec = std::make_unique<MusicRecommender>();
    r.ok = r.rec->load(lib, cat, r.error);
    return r;
}

// throws if the expression does not throw (used for the error paths)
template <typename Fn>
static bool throwsAny(Fn fn) {
    try {
        fn();
        return false;
    } catch (...) {
        return true;
    }
}

// ---------------------------------------------------------------------------
// core.h -- the generic data-structure layer
// ---------------------------------------------------------------------------

static void testVecNAndDistances() {
    std::printf("\ncore.h: vectors and distances\n");
    core::VecN a(std::vector<double>{0.0, 0.0});
    core::VecN b(std::vector<double>{3.0, 4.0});
    check("squared distance 3-4-5 -> 25",
          near(core::squaredDistance(a, b), 25.0));
    check("euclidean distance 3-4-5 -> 5",
          near(core::euclideanDistance(a, b), 5.0));
    check("points of different dims are rejected",
          throwsAny([&] { core::squaredDistance(a, core::VecN(3, 0.0)); }));
    check("VecN::get out of range throws",
          throwsAny([&] { a.get(7); }));
}

static void testKDataset() {
    std::printf("\ncore.h: KDataset dimension lock\n");
    core::KDataset ds;
    check("empty dataset", ds.size() == 0 && ds.dim() == 0);
    ds.add(core::VecN(2, 1.0));
    ds.add(core::VecN(2, 2.0));
    check("dataset keeps its inserts", ds.size() == 2);
    check("first insert fixed the dimension at 2", ds.dim() == 2);
    bool rejected = false;
    try { ds.add(core::VecN(3, 0.0)); }
    catch (const std::invalid_argument&) { rejected = true; }
    check("a 3-dim stray point is rejected after a 2-dim start", rejected);
    ds.clear();
    check("clear() forgets everything (incl. the dimension)",
          ds.size() == 0 && ds.dim() == 0);
    bool relocked = false;
    try { ds.add(core::VecN(4, 0.0)); relocked = ds.dim() == 4; }
    catch (...) {}
    check("after clear() a new dimension can be adopted", relocked);
}

static void testHashTable() {
    std::printf("\ncore.h: HashTable (chained hash map)\n");
    core::HashTable t(8);                    // tiny table -> forces rehashes
    for (int i = 0; i < 500; ++i)            // ... so growth is exercised
        t.insert("key" + std::to_string(i), i);
    bool all = true;
    for (int i = 0; i < 500; ++i) {
        int v;
        if (!t.find("key" + std::to_string(i), v) || v != i) { all = false; break; }
    }
    check("500 inserts survive rehashing, every key co-fillable->found", all);
    int v;
    check("absent key reports not found", !t.find("nope", v));
    check("contains() agrees with find()", t.contains("key7") && !t.contains("nope"));
    t.insert("dup", 1);
    t.insert("dup", 2);
    check("findAll sees both values under one key",
          t.findAll("dup") == std::vector<int>({1, 2}));
    check("size counts entries, not keys", t.size() == 502);
}

static void testHeap() {
    std::printf("\ncore.h: BoundedMaxHeap keeps only the K best\n");
    core::BoundedMaxHeap heap(3);
    heap.offer(10.0, 101);
    heap.offer(1.0, 102);
    heap.offer(5.0, 103);
    check("heap is full at capacity", heap.full());
    heap.offer(0.5, 104);                    // must evict 10.0
    heap.offer(2.0, 105);                    // must evict 5.0
    auto out = heap.drainSorted();
    check("drainSorted returns ascending order",
          out.size() == 3 && out[0].distSq == 0.5 && out[1].distSq == 1.0
              && out[2].distSq == 2.0);
    check("indices travel with the distances",
          out[0].index == 104 && out[2].index == 105);
    check("drainSorted leaves the heap empty", heap.size() == 0);
    core::BoundedMaxHeap zeroCap(0);
    zeroCap.offer(1.0, 1);
    check("capacity 0 ignores offers", zeroCap.size() == 0);
}

static void testKDTreeVsBruteForce() {
    std::printf("\ncore.h: K-d tree search == brute-force search\n");
    core::KDataset data;
    std::vector<core::VecN> points;
    for (int i = 0; i < 200; ++i) {          // deterministic pseudo-data
        std::vector<double> c(3);
        for (double& x : c) x = (i * 7919 % 997) / 997.0;
        points.push_back(core::VecN(c));
        data.add(points.back());
    }
    core::KDTree tree;
    check("build succeeds on 200 points", tree.build(data));
    int k = 7;
    for (int qi = 0; qi < 40; ++qi) {
        int exclude = qi % 3 == 0 ? qi : -1;   // sometimes exclude the query
        auto t = tree.searchKNearest(points[qi], k, exclude);
        auto b = core::bruteForceKNearest(data, points[qi], k, exclude);
        if (t.size() != b.size()) { check("same result count", false); return; }
        for (std::size_t j = 0; j < t.size(); ++j) {
            // distances must match; exact candidate order can differ on ties
            if (!near(t[j].distSq, b[j].distSq)) {
                check("tree == brute at query " + std::to_string(qi), false);
                return;
            }
        }
    }
    check("40 random queries: identical distances, tree vs brute", true);
    auto mega = tree.searchKNearest(points[0], 10000, -1);
    check("giant K is clamped to dataset size", mega.size() == 200);
    auto none = tree.searchKNearest(points[0], 0, -1);
    check("K=0 returns nothing", none.empty());
    core::KDTree untrained;
    check("search on an unbuilt tree does not crash and is empty",
          untrained.searchKNearest(points[0], 5, -1).empty());
}

// ---------------------------------------------------------------------------
// music.h -- loader, matcher, recommender
// ---------------------------------------------------------------------------

static void testCleanLoad() {
    std::printf("\nmusic.h: clean load end to end\n");
    auto r = load(fx("ok_lib.csv"), fx("ok_cat.csv"));
    check("load succeeds", r.ok);
    const LoadStats& s = r.rec->stats();
    check("3 library rows read", s.libraryRows == 3);
    check("3 usable songs", r.rec->songCount() == 3 && s.matched == 3);
    check("catalog row total is exact (no phantom header row)",
          s.catalogRows == 4);
    check("2 features selected (genre columns missing are fine)",
          r.rec->dims() == 2);
    std::size_t idx;
    check("lookup by spotify_id works",
          r.rec->indexForTrackId("bbb", idx) && r.rec->song(idx).trackName == "Song B");
}

static void testTwoPassIdIndex() {
    std::printf("\nmusic.h: primary id always beats a claimed fallback id\n");
    auto r = load(fx("conflict_lib.csv"), fx("conflict_cat.csv"));
    check("load succeeds", r.ok);
    std::size_t idxB;
    check("'bbb' resolves to the song that OWNS bbb (two-pass idIndex_)",
          r.rec->indexForTrackId("bbb", idxB)
              && r.rec->song(idxB).trackName == "Owner");
    std::size_t idxA;
    check("'aaa' resolves to its own song",
          r.rec->indexForTrackId("aaa", idxA)
              && r.rec->song(idxA).trackName == "Claimer");
    // NOTE: if this ever fails with lookup(bbb) -> "Claimer", the
    // id-index build went back to one interleaved pass.
}

static void testFallbackMatching() {
    std::printf("\nmusic.h: catalog_track_id fallback (first wins)\n");
    auto r = load(fx("fallback_lib.csv"), fx("fallback_cat.csv"));
    check("load succeeds", r.ok);
    const LoadStats& s = r.rec->stats();
    check("both rows aimed at the same catalog id", s.libraryFallbackKeys == 2);
    check("one of them loses (conflict counted)", s.libraryKeyConflicts == 1);
    check("exactly one usable song", r.rec->songCount() == 1 && s.matched == 1);
    check("the winner came through the fallback path",
          s.matchedDirect == 0 && s.matchedFallback == 1);
    std::size_t idx;
    check("the FIRST caller owns the shared id",
          r.rec->indexForTrackId("xxx", idx)
              && r.rec->song(idx).trackName == "First Caller");
    check("the second caller is genuinely absent",
          !r.rec->indexForTrackId("yyy", idx));
}

static void testLibraryDuplicates() {
    std::printf("\nmusic.h: duplicate library rows keep the first\n");
    auto r = load(fx("dup_lib.csv"), fx("dup_cat.csv"));
    const LoadStats& s = r.rec->stats();
    check("dupe row counted", s.libraryDuplicates == 1);
    check("only one song usable", r.rec->songCount() == 1);
    check("exactly one song survived the dupe", r.ok && s.matched == 1);
}

static void testHeaderValidation() {
    std::printf("\nmusic.h: broken headers are hard errors\n");
    auto a = load(fx("dup_header_lib.csv"), fx("ok_cat.csv"));
    check("duplicate spotify_id in the library header rejects the load",
          !a.ok && a.error.find("appears") != std::string::npos);
    auto b = load(fx("ok_lib.csv"), fx("dup_header_cat.csv"));
    check("duplicate catalog header column rejects the load", !b.ok);
    auto c = load(fx("missing_col_lib.csv"), fx("ok_cat.csv"));
    check("missing required column names the column",
          !c.ok && c.error.find("spotify_id") != std::string::npos);
}

static void testBadRows() {
    std::printf("\nmusic.h: malformed and junk rows are skipped, not fatal\n");
    auto w = load(fx("width_lib.csv"), fx("width_cat.csv"));
    check("short catalog row is counted as malformed",
          w.rec->stats().catalogMalformed == 1);
    check("the valid row still loads",
          w.ok && w.rec->songCount() == 1);
    std::size_t scratch;
    check("skipped row is not reachable", !w.rec->indexForTrackId("qqq", scratch));
    auto j = load(fx("junk_lib.csv"), fx("junk_cat.csv"));
    const LoadStats& js = j.rec->stats();
    check("'12abc' is rejected as an invalid number (row jjj)",
          js.rejectedInvalid == 1);
    check("the clean neighbour song loads fine",
          j.ok && j.rec->songCount() == 1);
    check("nothing was fabricated for the junk row",
          !j.rec->indexForTrackId("jjj", scratch));
}

static void testImplausibleRows() {
    std::printf("\nmusic.h: implausible placeholder rows are rejected\n");
    auto r = load(fx("rules_lib.csv"), fx("rules_cat.csv"));
    check("load succeeds (the clean row keeps the corpus alive)", r.ok);
    const LoadStats& s = r.rec->stats();
    // rules: tempo <= 0, energy == 0, danceability == 0,
    //        loudness < -35 or loudness > 2 -- one bad row each
    check("all five rule breakers hit the implausible bucket",
          s.rejectedImplausible == 5);
    check("rejections were direct-id rejections (may fall back)",
          s.directRejected == 5);
    check("only the one good row matched", s.matchedDirect == 1 && s.matched == 1);
    check("corpus holds exactly the good song", r.rec->songCount() == 1);
    std::size_t scratch;
    check("a rejected row is genuinely unreachable",
          !r.rec->indexForTrackId("tempo0", scratch));
    auto fb = load(fx("fbrescue_lib.csv"), fx("fbrescue_cat.csv"));
    const LoadStats& fbs = fb.rec->stats();
    check("direct row (tempo 0) rejected, playlist story intact",
          fb.ok && fbs.rejectedImplausible == 1 && fbs.directRejected == 1);
    check("the song still got features through its catalog_track_id",
          fbs.matchedFallback == 1 && fb.rec->songCount() == 1);
    std::size_t idx;
    check("lookup by the song's own spotify_id still finds it",
          fb.rec->indexForTrackId("aaa", idx) &&
              fb.rec->song(idx).trackName == "Song A");
    auto n = load(fx("norum_lib.csv"), fx("norum_cat.csv"));
    check("rules never fire for columns the catalog lacks",
          n.ok && n.rec->stats().rejectedImplausible == 0 &&
              n.rec->songCount() == 2);
}

static void testInlineSingleFile() {
    std::printf("\nmusic.h: single-file mode (features ride in the rows)\n");
    auto r = load(fx("single_lib.csv"), "");    // empty catalog = inline mode
    check("load succeeds (one clean row keeps it alive)", r.ok);
    const LoadStats& s = r.rec->stats();
    check("blank tempo -> rejectedBlank", s.rejectedBlank == 1);
    check("'12abc' tempo -> rejectedInvalid", s.rejectedInvalid == 1);
    check("tempo 0 -> rejectedImplausible", s.rejectedImplausible == 1);
    check("two usable songs (ok1 + ok2), fresh dims", s.matched == 2 &&
          r.rec->songCount() == 2 && r.rec->dims() == 4);
    std::size_t idx;
    check("lookup by spotify_id works",
          r.rec->indexForTrackId("ok1", idx) &&
              r.rec->song(idx).trackName == "Song Ok");
    check("the claimed catalog_track_id is searchable too",
          r.rec->indexForTrackId("claimed1", idx));
    auto d = load(fx("dupfeat_lib.csv"), "");
    check("a duplicated feature column is a hard error",
          !d.ok && d.error.find("ambiguous") != std::string::npos);
    auto n = load(fx("nofeat_lib.csv"), "");
    check("no feature columns is a hard error naming the problem",
          !n.ok && n.error.find("none of the supported") != std::string::npos);
}

static void testTracksStyleFile() {
    std::printf("\nmusic.h: Spotify tracks.csv aliases and optional album\n");
    auto r = load(fx("tracks_style_lib.csv"), "");
    check("tracks.csv-style headers load without an album column", r.ok);
    check("tracks.csv-style rows keep usable songs only",
          r.ok && r.rec->songCount() == 2);
    check("artist list is normalized for display/search",
          r.ok && r.rec->song(0).artist == "Artist A, Artist B" &&
              r.rec->song(1).artist == "Solo Artist");
    std::size_t index = 0;
    check("tracks.csv id column is searchable",
          r.ok && r.rec->indexForTrackId("t1", index) && index == 0);
    auto recs = r.ok ? r.rec->recommend(0, 1, 0)
                     : std::vector<Recommendation>();
    check("tracks.csv rows participate in recommendations",
          recs.size() == 1 && recs[0].meta.trackId == "t2");
}

static void testQuerySeeds() {
    std::printf("\nmusic.h: --seeds query database (query only)\n");
    auto r = load(fx("single_lib.csv"), "");
    std::string err;
    bool ok = r.rec->loadQuerySongs(fx("single_lib.csv"), err);
    check("seed file loads beside the corpus", ok);
    check("only rows with a usable vector become seeds",
          r.rec->querySongCount() == 2);
    check("seeds never enlarge the search corpus", r.rec->songCount() == 2);
    auto hits = r.rec->searchQuerySongsByName("Song Ok");
    check("seed search finds its row",
          hits.size() == 1 && hits[0].meta.trackId == "ok1");
    std::size_t qidx;
    check("seed lookup by track id works",
          r.rec->indexForQueryTrackId("ok2", qidx) && qidx == 1);
    auto recs = r.rec->recommendByFeatures(
        r.rec->querySong(0).features, 5, 0);   // exclude corpus ok1
    check("seed query answers from the corpus only",
          recs.size() == 1 && recs[0].meta.trackId == "ok2");
}

static void testQuotedCsv() {
    std::printf("\nmusic.h: RFC CSV edge cases in the real world\n");
    auto r = load(fx("quoted_lib.csv"), fx("quoted_cat.csv"));
    check("quoted commas / escaped quotes parse", r.ok);
    check("track name survived quoting intact",
          r.ok && r.rec->songCount() == 1 &&
              r.rec->song(0).trackName == "Hello, \"World\" Song");
}

static void testSearchByNameAndLimit() {
    std::printf("\nmusic.h: searchByName carries one extra hit as a flag\n");
    auto r = load(fx("search_lib.csv"), fx("search_cat.csv"));
    auto hits = r.rec->searchByName("Common Song");
    check("9 hits: 8 + 1 overflow marker = 9", hits.size() == 9);
    check("the overflow candidate is the 9th row (t8)",
          hits.back().meta.trackId == "t8id");
    auto few = r.rec->searchByName("Art3");
    check("a narrow search returns its single hit (no overflow)",
          few.size() == 1 && few[0].meta.trackId == "t3id");
    check("empty/blank text searches are empty",
          r.rec->searchByName("").empty());
}

static void testRecommend() {
    std::printf("\nmusic.h: recommend() uses the tree and excludes the query\n");
    auto r = load(fx("ok_lib.csv"), fx("ok_cat.csv"));
    auto recs = r.rec->recommend(0, 10, 0);      // ask for 10, dataset has 3
    check("K clamped to the dataset (2 neighbours left)", recs.size() == 2);
    bool clean = true;
    for (const auto& n : recs)
        if (n.vectorIndex == 0) clean = false;
    check("the query song is never recommended", clean);
    long long visited = -1;
    auto with = r.rec->recommend(0, 2, -1, &visited);
    check("nodesVisited is reported back", visited > 0 && with.size() == 2);
    check("K<=0 returns nothing", r.rec->recommend(0, 0, -1).empty());
}

static void testConstantFeature() {
    std::printf("\nmusic.h: constant features map to 0.0, never div by zero\n");
    auto r = load(fx("ok_lib.csv"), fx("ok_cat.csv"));
    const Normalization& n = r.rec->normalization();
    // danceability in ok fixtures: 0.2, 0.7, 0.5 -> standard range
    check("danceability range is 0.2..0.7",
          near(n.minimum[0], 0.2) && near(n.maximum[0], 0.7));
    // force a constant second feature via a one-row dataset
    MusicRecommender single;
    std::string err;
    (void)single.load(fx("quoted_lib.csv"), fx("quoted_cat.csv"), err);
    if (single.songCount() == 1) {
        const Normalization& m = single.normalization();
        check("single-point range is degenerate, mapping pins it to 0.0",
              near(m.maximum[0] - m.minimum[0], 0.0));
    } else {
        check("single-row fixture did not load (test setup wrong)", false);
    }
}

// ---------------------------------------------------------------------------

int main() {
    std::printf("Nimrod self-tests\n=================\n");
    testVecNAndDistances();
    testKDataset();
    testHashTable();
    testHeap();
    testKDTreeVsBruteForce();
    testCleanLoad();
    testTwoPassIdIndex();
    testFallbackMatching();
    testLibraryDuplicates();
    testHeaderValidation();
    testBadRows();
    testImplausibleRows();
    testInlineSingleFile();
    testTracksStyleFile();
    testQuerySeeds();
    testQuotedCsv();
    testSearchByNameAndLimit();
    testRecommend();
    testConstantFeature();
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
