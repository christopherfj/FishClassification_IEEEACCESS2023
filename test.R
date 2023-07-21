library(readr)

test <- read_csv("test.csv")
metrics <- c("accuracy", "precision", "recall", "f1")
alpha <- 0.05
total <- 0
count <- 0

for(metric in metrics){
  keys <- names(test)
  keys <- keys[grepl(paste("_", metric, sep=""), keys)]
  keyA = paste("dl_DL_BOTH_", metric, sep="") 
  keys <- keys[keys!=keyA]
  A <- test[[keyA]]
  for(keyB in keys){
    total <- total+1
    B = test[[keyB]]
    result = wilcox.test(A, B)
    pvalue <- result["p.value"]
    if(pvalue<alpha){
      print( paste(keyA, keyB, metric, "p<0.05", sep="-") )
      count <- count+1
    }
  }  
}

print(100*count/total)