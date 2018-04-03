
library(PASWR2)
set.seed(12)
df <- TITANIC3
df <- df %>% 
  select(age, cabin, fare, name, parch, pclass, sex, sibsp, survived, ticket, embarked) %>% 
  rename(Age=age, Cabin=cabin, Fare=fare, Name=name, Parch=parch, Pclass=pclass, Sex=sex, SibSp=sibsp, Survived=survived, Ticket=ticket, Embarked=embarked) %>% 
  sample_frac() 
df$Data <- 'train'
df$Data[892:dim(df)[1]] <- 'test' 
df$PassengerId <- row.names(df)

y_test <- df %>% 
  filter(Data=='test') %>% 
  select(Survived, PassengerId)

df$Survived[df$Data=='test'] <- factor(NA)   

write.csv(df, file = "/home/troels/git/danske_tech/titanic.csv", row.names = FALSE)
write.csv(y_test, file = "/home/troels/git/danske_tech/y_test.csv", row.names = FALSE)