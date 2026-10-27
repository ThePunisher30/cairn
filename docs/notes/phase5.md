# Phase 5: checkpoint notes

The numbers quoted here come from running the real code (see `docs/results.md`, section "Phase 5"). The
100,000-vector set is synthetic (mixes of real vectors plus noise); "real" means the 3,267 chunk vectors.

## 1. Why can't you build a perfect index for nearest-neighbour search in high dimensions? What does your own data say about it?

In 2 or 3 dimensions you can cut space into boxes, and for a query most boxes are provably too far away to hold
a nearest neighbour, so you skip them and stay exact. In 384 dimensions almost every vector is about equally far
from almost every other. A box (or a list, or a graph neighbourhood) that contains the true neighbour is not
much closer than the ones next to it, so nothing can be skipped safely, and the exact method turns into a full
scan. That is why the indexes I built are approximate: they skip things that are only *probably* far.

My data shows it as a price for the last bit of recall:
- IVF (`nlist` 316, 100,000 vectors): recall 0.953 costs 1,207 distances, 0.995 costs 3,731, and 1.000 costs
  18,653 (`nprobe` 48): the last 0.005 costs 5 times more than the first 0.95.
- IVF with `nlist` 100 needs `nprobe` 64 for recall 1.000, which is 66,936 distances, two thirds of a full scan.
- HNSW stops at 0.999 for `ef` 160 and 320 (and for `M` 32): 2 of 2,000 true neighbours were never found. I
  don't know why.
- On the real corpus IVF at `nprobe` 64 (more than the 57 lists, so all of them) computes 3,324 distances, more than the flat
  scan's 3,267, to get 1.000.

## 2. In IVF, what does raising `nprobe` do to recall, speed and memory? What does raising `nlist` do? Why is about the square root of `n` a sensible `nlist`?

`nprobe` is a search-time setting. Raising it scans more lists: recall goes up (`nlist` 316: 0.868, 0.953, 0.992,
0.995, 1.000 for `nprobe` 1, 2, 6, 8, 48), speed goes down (760 to 18,653 distances), and memory does not change,
because the index is the same.

Raising `nlist` makes the lists shorter, so each probe is cheaper, but every search first scores all `nlist`
centres, and the index is bigger and slower to build (19.6, 30.7, 54.5 s and 0.95, 1.29, 2.34 MB for `nlist` 100,
316, 1,000). With too few lists a probe is expensive: `nlist` 100 needed 3 to 7 times more distances than the others.

A search costs about `nlist + nprobe x s x n / nlist`: the centre scores plus the vectors in the probed lists.
`s` is about 1.4 to 1.5 because the lists are uneven (at `nlist` 316 `nprobe` 1 scans 444 vectors where an even
split would give 316). The first term grows with `nlist` and the second shrinks, and the sum is smallest where
they are equal in size: `nlist = sqrt(nprobe x s x n)`. For `nprobe` 1 and `n` = 100,000 that is 374, close to
sqrt(n) = 316. In the measurements 316 was the best at recall 0.95 (1,207 distances against 1,301 for 1,000), and
1,000 was the best at 0.99 (2,156 against 2,908), where `nprobe` is 8 and the formula gives 1,058. So the square
root is a good start for small `nprobe`, and the best `nlist` rises with the recall wanted.

## 3. What does k-means++ do that random starting centres do not, and what goes wrong with an empty cluster?

Random starting centres are often close together: two land in the same real group and another group gets none,
and k-means then keeps that bad split for the whole run. k-means++ picks the first centre at random and each next
one with probability proportional to its squared distance from the nearest centre already chosen (for unit
vectors `2 - 2 x dot`), so new centres tend to fall in places the old ones do not cover. I did not measure the
difference on this data.

An empty cluster is a centre that no vector is nearest to. Its new position would be the average of nothing, a
zero vector, and a zero vector has a zero dot product with everything: it would never win again, the index would
have one list fewer than `nlist`, and normalising it divides by zero. My k-means moves such a centre onto the
vector that fits its current centre worst, so the cluster is used again and the vector that was the least
well-represented gets a centre of its own.

## 4. In HNSW, what does raising `efSearch` do? What does raising `M` do? What does raising `efConstruction` do, and who pays for it?

- `ef_search` is the beam width of a search. Raising it keeps more candidates, so recall rises and the search
  does more work (`M` 16: 0.921 at 231 distances for `ef` 10, 0.993 at 695 for `ef` 80, 0.999 at 2,479 for
  `ef` 320). The searcher pays, per query.
- `M` is the number of links per vector (`2M` on layer 0). More links give a better-connected graph and more
  recall at the same `ef`, and cost memory and build time and a little more work per step: `M` 8, 16, 32 gave
  0.975, 0.993, 0.992 at `ef` 80, for 6.3, 11.4, 21.5 MB and 269, 482, 917 s. `M` 16 was the best; 32 gave no
  more recall for twice the memory.
- `ef_construction` is the beam width while building, which decides how good the chosen neighbours are. 40 gave a
  clearly worse graph (0.954 at `ef` 80), 100 gave 0.993, 200 gave 0.987 and took 1.44 times as long to build.
  The builder pays, once, in build time; the benefit is for every search. Beyond 100 I found no benefit.

## 5. Why are there layers? What would a search be like without the upper layers?

A walk on a graph where every vector links only to near neighbours makes small steps, so from a far-away start it
takes many steps to reach the query's neighbourhood. The upper layers hold fewer and fewer vectors (3,267 on layer
0, 209 on layer 1, 10 on layer 2 and 1 on layer 3 in one real graph) with links that span longer distances. A
search starts at the one vector on the top layer, walks greedily down, and arrives at layer 0 already close.

Without them the layer-0 beam starts at the entry vector. I measured it on the same graphs: at 100,000 vectors
the layers gave 0.025 more recall and 15% fewer distances at `ef` 10 (0.921 against 0.896, 231 against 272); at
`ef` 160 there was no difference. At 5,000 vectors and on the real corpus there was no benefit (on the real
corpus skipping them was slightly better at small `ef` in all three graphs, but 40 queries and the seed-to-seed
spread of 0.065 mean I read it as "no difference"). So the answer to "what would a search be like without them" is
"the same, at this size, and a little worse and more expensive at 100,000 with a small `ef`". The tests cannot
tell the two apart, so only measuring shows it.

## 6. Why does choosing "the `M` closest" as neighbours go wrong on clustered data, and what does the heuristic change?

If vectors lie in tight clusters, the `M` closest to a vector are all in its own cluster. The links then stay
inside the cluster and no link leads out to another cluster, so a walk that starts elsewhere or has to cross
over has no road to the target. The heuristic goes through the candidates from the closest and keeps one only
if it is closer to the new vector than to any neighbour already kept; the others fill free slots at the end. A
candidate in the same cluster as an earlier pick is closer to that pick than to the new vector, so it is
skipped, and the kept links point in different directions, including out of the cluster.

The measured size on the 100,000-vector set (clustered by construction): recall 0.993 with it against 0.913
without at `ef` 80, and 0.999 against 0.969 at `ef` 320; without it recall never reached 0.99. The heuristic costs
build time (482 s against 198 s). I did not test non-clustered data, and I did not check directly that the
missing links between clusters are the cause.

## 7. Why does your corpus not need ANN? At what size would you start to need it, from your own numbers?

The flat index searches the 3,267 real vectors in 0.36 ms (2,795 queries/s). IVF only matched that at recall 0.94
(2,961 queries/s) and HNSW at recall 0.993 was 3 times slower (959). Searching is not the cost of a query anyway:
embedding the question takes 38.5 ms, about 100 times the scan.

From the scale set, a flat scan of 100,000 vectors takes about 6.7 ms (150 queries/s). If it is linear in `n`
(extrapolated, not measured), it equals the 38.5 ms of embedding at about 600,000 vectors. I would start to
consider ANN at a few hundred thousand chunks, or earlier for many simultaneous queries, and test again at that
size. Memory is the other limit: 153.6 MB per 100,000 vectors of 384 numbers.

## 8. What did your pure-Python HNSW build cost, and what does that tell you about why real libraries are written in C++?

482 s for 100,000 vectors (`M` 16, `ef_construction` 100), 4.8 ms per inserted vector; 917 s for `M` 32 and 13 s for
the 3,267 real vectors. IVF with 316 lists built in 31 s. Every insertion runs a beam search that computes
hundreds of dot products, and the loop around them (heaps, a visited set, choosing neighbours) is Python,
with numpy called on a few dozen vectors at a time, where it cannot do its best. The same is true of a search:
HNSW computes 144 to 179 times fewer distances than flat at recall 0.99 but is only 3 to 5 times faster, because
the distances are cheap and the Python around them is not. A C++ library does the loops, the heaps and the
distance code compiled and can use many cores for the build. I did not profile my code, so where exactly the time
goes is my explanation, not a measurement.
