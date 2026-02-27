# ChessZero
A fully vectorized AlphaZero implementation for Chess using JAX, Flax, mctx, and Pgx.

For installing BayesElo program, run these commands in the project root:
```
curl -O https://www.remi-coulom.fr/Bayesian-Elo/bayeselo.tar.bz2
tar -xjf bayeselo.tar.bz2
cd BayesElo
make
mv bayeselo ../
cd ..
rm -rf bayeselo.tar.bz2 BayesElo
```