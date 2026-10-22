# Resume entry

**Drape — Personalised Fashion Search & Recommendation Engine** | Python, NumPy, SciPy, FastAPI, Docker | github.com/mrperfectme01/drape

- Built an end-to-end e-commerce search and recommendation system over 20K products and 563K user events. Every core algorithm was implemented from scratch (BM25F, LSA, HNSW, implicit ALS, LambdaRank) with no ML libraries.
- Raised search NDCG@10 from 0.06 to 0.44 (7.3×) and purchase MRR 8.8× over BM25 with hybrid sparse+dense retrieval, a learned query classifier and a personalised LambdaRank MLP over 16 features.
- Improved home-feed Recall@20 by 42% over matrix factorisation with multi-source candidate generation (ALS via MIPS→HNSW, session co-visitation, popularity) and a second learning-to-rank stage.
- Implemented an HNSW ANN index reaching 0.98 recall@10 with 23× fewer distance computations than exact search. Built real-time personalisation through ALS fold-in (<1 ms per event).
- Designed a leakage-free time-split evaluation. Caught and regression-tested a float32 timestamp bug that leaked same-session clicks into 49% of offline queries.
- Trained an outfit-compatibility model that rediscovered all 7 planted colour-harmony rules from co-purchases alone (AUC 0.89). Served everything through FastAPI at 8–18 ms p50 with caching, latency metrics and 22 tests.

*Pick 3–4 bullets for a one-page resume. Bullets 2, 4 and 5 are the strongest for SDE and ML roles.*

---

# Interview prep: questions you will get

**Why does personalisation help search this much?**
Most fashion queries ("black shorts") match hundreds of products that are nearly identical as text.
BM25 can't separate them. The deciding signal is the shopper's style, colour and price preference.
That is why `user_style_aff` and `als_score` are the top ranker features.

**Why LambdaRank and not pointwise regression?**
Ranking metrics only care about order at the top. LambdaRank weights each pairwise gradient by
|ΔNDCG| of swapping the pair, so the model spends capacity on errors in positions 1–10, not position 200.

**Why ALS with conjugate gradient?**
An exact solve is O(k³) per user. CG with 3 warm-started steps is O(nnz·k), which is what the
`implicit` library does. Fold-in is a single 64×64 solve, so new clicks update the user vector without retraining.

**How does HNSW work? Why is it slower than brute force here?**
It is a hierarchy of proximity graphs: greedy descent through sparse upper layers, then beam
search with width ef on the dense bottom layer. It computes 879 distances against 20,000 for a scan,
but each graph hop is a Python operation while brute force is one BLAS call. At millions of items
the log-scaling wins. In production I'd use FAISS or ScaNN.

**How do you do inner-product search with an L2 index?**
Append √(M² − ‖x‖²) to every item and 0 to the query. Then ‖q − x‖² = const − 2⟨q, x⟩, so the nearest neighbour is the max inner product.

**How did you avoid leakage?**
Rankers train on a window after the models that produce their features were frozen. Test uses a
later window. Session context comes only from events strictly before the query. Then tell the
float32 story: how you noticed it, how you measured it (49%), how you fixed it (guard band),
and how a test stops it coming back.

**Your data is simulated. Why trust the results?**
Real datasets lack queries plus sessions plus outfits together. The simulator has known ground
truth, so every metric is verifiable. The outfit model recovering the planted colour rules shows
the pipeline learns real structure. The relative ordering of methods is what transfers. Next step: the H&M Kaggle dataset.

**How would you scale it?**
Move user state to Redis, run stateless workers, use FAISS for ANN, run nightly batch retraining of
ALS and the rankers, stream events through Kafka, and replace offline metrics with interleaving or A/B tests.
