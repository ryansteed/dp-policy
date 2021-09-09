library(stargazer)

df = read.csv("results/discrimination_laplace.csv")

lm = lm(error*1e6 ~ race_1est + race_2est, data=df)
summary(lm)

lm2 = lm(abs(error)*1e6 ~ race_1est + race_2est, data=df)
summary(lm2)

stargazer(
  lm, lm2, 
  type="latex",
  title="Correlation between race and noise-induced error.",
  covariate.labels=c("Total individuals", "White-only individuals"),
  label="discrimination",
  out="results/discrimination.tex"
)
