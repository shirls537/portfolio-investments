analyst = "Shirley Wang" # Replace this with your name

f = "setup.R"; for (i in 1:10) { if (file.exists(f)) break else f = paste0("../", f) }; source(f)
options(repr.matrix.max.rows=674)
options(repr.matrix.max.cols=200)
update_geom_defaults("point", list(size=1))                                

# Set the business parameters.

budget = 1000000
portfolio_size = 12

data.frame(budget, portfolio_size)

# Set any additional business parameters.
allocation = rep(budget/portfolio_size, portfolio_size)
fmt(allocation)

# Retrieve "My Data.csv"
#   ... OR ...
# Retrieve "Company Fundamentals 2017.csv"

# This is the ORIGINAL data.
data = read.csv('My Data.csv', header=TRUE)
data$big_growth = factor(data$big_growth, levels=c("YES", "NO"))

head(data)

# Outome variable: big_growth
# Model 1 SVM predictor variables: PC1, PC2, PC3
# Model 2 SVM predictor variables: PC3
# Model 3 SVM predictor variables: PC1, PC2

set.seed(12345)
model.1 = svm(big_growth ~ PC1+PC2+PC3, data, type="C-classification", kernel="polynomial", degree=2, cost=1, scale=TRUE, probability=TRUE)
set.seed(12345)
model.2 = svm(big_growth ~ PC3, data, type="C-classification", kernel="polynomial", degree=3, cost=10, scale=TRUE, probability=TRUE)
set.seed(12345)
model.3 = svm(big_growth ~ PC1+PC2, data, type="C-classification", kernel="radial", gamma=1, cost=100, scale=TRUE, probability=TRUE)

prob.1 = attr(predict(model.1, data, probability=TRUE), "probabilities")
class.predicted.1 = as.class(prob.1, "YES", cutoff=0.25)

prob.2 = attr(predict(model.2, data, probability=TRUE), "probabilities")
class.predicted.2 = as.class(prob.2, "YES", cutoff=0.25)

prob.3 = attr(predict(model.3, data, probability=TRUE), "probabilities")
class.predicted.3 = as.class(prob.3, "YES", cutoff=0.25) 

data.stack.1 = data.frame(data, class.predicted.1, class.predicted.2, class.predicted.3)
head(data.stack.1)

# Experimenting with Neural Networks to predict big_growth
# Outcome variable: big_growth (binary)
# Predictor variables: PC1, PC2, PC3

set.seed(12345)

data$big_growth.bin = as.binary(data$big_growth, "YES")
model = neuralnet(big_growth.bin ~ PC1+PC2+PC3, data, hidden=c(2,2,2), algorithm="rprop+", act.fct="logistic", linear.output=FALSE, rep=1)

model


model$weights[[1]]
output_size(5,5)
plot(model, rep=1, fill="gray", show.weights=TRUE, information=FALSE, cex=0.7, lwd=0.5, arrow.length=0.15)
output_size(restore)

# Transform representation of data, if necessary


# Construct a model to predict big_growth or growth.
# Present a brief summary of the model parameters.

# Experimenting with SVM to predict big_growth, applying with boosting method
# Outcome Variable: big_growth
# Predictor Variables: PC1, PC2, PC3

set.seed(12345)
model.1 = svm(big_growth ~ PC1+PC2+PC3, data, type="C-classification", kernel="polynomial", degree=3, cost=10, scale=TRUE, probability=TRUE)
prob.1 = attr(predict(model.1, data, probability=TRUE), "probabilities")
big_growth.predicted.1 = as.class(prob.1, "YES", cutoff=0.5)
hit.1 = big_growth.predicted.1 == data$big_growth

result.1 = data.frame(data, big_growth.predicted.1, hit.1)
head(result.1)

nrow(result.1[result.1$hit.1==TRUE,])

set.seed(12345)
data.2 = focus_data(data, hit.1, emphasis=10)
head(data.2)

set.seed(12345)
model.2 = svm(big_growth ~ PC1+PC2+PC3, data.2, type="C-classification", kernel="polynomial", degree=3, cost=10, scale=TRUE, probability=TRUE)

prob.2 = attr(predict(model.2, data.2, probability=TRUE), "probabilities")
big_growth.predicted.2 = as.class(prob.2, "YES", cutoff=0.5)
hit.2 = big_growth.predicted.2 == data.2$big_growth

result.2 = data.frame(data.2, big_growth.predicted.2, hit.2)
head(result.2)

# Calculate and present the model's estimated profit and profit rate.
k=5
set.seed(0)
fold = createFolds(data$big_growth, k=k)
str(fold)

fold_num = c()
profit = c()

for (iteration in 1:5) {
    fold_num[iteration] = iteration
    data.test  = data[,2:9][fold[[iteration]],]
    data.train = data[,2:9][setdiff(1:nrow(data), fold[[iteration]]),]
    
    set.seed(12345)
    model.1 = svm(big_growth ~ PC1+PC2+PC3, data.train, type="C-classification", kernel="polynomial", degree=2, cost=1, scale=TRUE, probability=TRUE)
    set.seed(12345)
    model.2 = svm(big_growth ~ PC3, data.train, type="C-classification", kernel="polynomial", degree=3, cost=10, scale=TRUE, probability=TRUE)
    set.seed(12345)
    model.3 = svm(big_growth ~ PC1+PC2, data.train, type="C-classification", kernel="radial", gamma=1, cost=100, scale=TRUE, probability=TRUE)

    prob.1 = attr(predict(model.1, data.train, probability=TRUE), "probabilities")
    class.predicted.1 = as.class(prob.1, "YES", cutoff=0.25)

    prob.2 = attr(predict(model.2, data.train, probability=TRUE), "probabilities")
    class.predicted.2 = as.class(prob.2, "YES", cutoff=0.25)

    prob.3 = attr(predict(model.3, data.train, probability=TRUE), "probabilities")
    class.predicted.3 = as.class(prob.3, "YES", cutoff=0.25) 

    data.stack.1 = data.frame(data.train, class.predicted.1, class.predicted.2, class.predicted.3)
    
    prob.1 = attr(predict(model.1, data.test, probability=TRUE), "probabilities")
    class.predicted.1 = as.class(prob.1, "YES", cutoff=0.25)

    prob.2 = attr(predict(model.2, data.test, probability=TRUE), "probabilities")
    class.predicted.2 = as.class(prob.2, "YES", cutoff=0.25)

    prob.3 = attr(predict(model.3, data.test, probability=TRUE), "probabilities")
    class.predicted.3 = as.class(prob.3, "YES", cutoff=0.25)

    data.stack.2 = data.frame(data.test, class.predicted.1, class.predicted.2, class.predicted.3)
    data.stack = data.stack.1[,c('big_growth','class.predicted.1','class.predicted.2','class.predicted.3')]
    model.stack = naiveBayes(big_growth~class.predicted.1+class.predicted.2+class.predicted.3, data.stack)
    output.predicted = predict(model.stack, data.stack.2[,c('class.predicted.1','class.predicted.2','class.predicted.3')], type="raw")
    prob.stack = data.frame("YES"=output.predicted, "NO"=1-output.predicted)[,c(1:2)]
    names(prob.stack) = c("YES","NO")
    
    data.test$prob_yes = prob.stack[,1]
    
    top_twelve = data.test[order(-data.test$prob_yes),][1:12,]
    growth_allocation = 0
    
    for (i in 1:12) {
        growth_allocation = growth_allocation + (1+top_twelve$growth[i])*allocation[i]
    }
    profit[iteration] = growth_allocation - budget
}

five_fold.cv = data.frame(fold_num, profit)
five_fold.cv

profit_rate.cv = mean(five_fold.cv$profit/budget)
fmt(profit_rate.cv)

fold_num = c()
accuracy = c()
profit = c()

for (iteration in 1:5) {
    fold_num[iteration] = iteration
    data.test  = data[,2:9][fold[[iteration]],]
    data.train = data[,2:9][setdiff(1:nrow(data), fold[[iteration]]),]
    
    data.train$big_growth.bin = as.binary(data.train$big_growth, "YES")
    model = neuralnet(big_growth.bin ~ PC1+PC2+PC3, data.train, hidden=c(2,2,2), algorithm="rprop+", act.fct="logistic", linear.output=FALSE, rep=1)
    
    output = compute(model, data.test, rep=1)$net.result
    prob = data.frame("YES"=output, "NO"=1-output)
    class.predicted = as.class(prob, "YES", 0.5)

    data.test$big_growth.predicted = class.predicted
    data.test$prob_yes = prob[,1]
    CM = confusionMatrix(data.test$big_growth.predicted, data.test$big_growth)$table
    cm = CM/sum(CM)
    accuracy[iteration] = cm[1,1]+cm[2,2]
    
    
    top_twelve = data.test[order(-data.test$prob_yes),][1:12,]
    growth_allocation = 0
    
    for (i in 1:12) {
        growth_allocation = growth_allocation + (1+top_twelve$growth[i])*allocation[i]
    }
    profit[iteration] = growth_allocation - budget
}

five_fold.cv = data.frame(fold_num, accuracy, profit)
five_fold.cv

profit_rate.cv = mean(five_fold.cv$profit/budget)
fmt(profit_rate.cv)

fold_num = c()
accuracy = c()
profit = c()

for (iteration in 1:5) {
    fold_num[iteration] = iteration
    data.test  = data[,2:9][fold[[iteration]],]
    data.train = data[,2:9][setdiff(1:nrow(data), fold[[iteration]]),]
    
    set.seed(12345)
    model.1 = svm(big_growth ~ PC1+PC2+PC3, data.train, type="C-classification", kernel="polynomial", degree=3, cost=10, scale=TRUE, probability=TRUE)
    prob.1 = attr(predict(model.1, data.train, probability=TRUE), "probabilities")
    big_growth.predicted.1 = as.class(prob.1, "YES", cutoff=0.5)
    hit.1 = big_growth.predicted.1 == data.train$big_growth

    result.1 = data.frame(data.train, big_growth.predicted.1, hit.1)
    
    set.seed(12345)
    data.train.2 = focus_data(data.train, hit.1, emphasis=10)
    
    set.seed(12345)
    model.2 = svm(big_growth ~ PC1+PC2+PC3, data.train.2, type="C-classification", kernel="polynomial", degree=3, cost=10, scale=TRUE, probability=TRUE)
    
    prob.2 = attr(predict(model.2, data.test, probability=TRUE), "probabilities")
    big_growth.predicted.2 = as.class(prob.2, "YES", cutoff=0.5)
    
    data.test$big_growth.predicted = big_growth.predicted.2
    data.test$prob_yes = prob.2[,1]

    CM = confusionMatrix(data.test$big_growth.predicted, data.test$big_growth)$table
    cm = CM/sum(CM)
    accuracy[iteration] = cm[1,1]+cm[2,2]
    
    
    top_twelve = data.test[order(-data.test$prob_yes),][1:12,]
    growth_allocation = 0
    
    for (i in 1:12) {
        growth_allocation = growth_allocation + (1+top_twelve$growth[i])*allocation[i]
    }
    profit[iteration] = growth_allocation - budget
}

five_fold.cv = data.frame(fold_num, accuracy, profit)
five_fold.cv

profit_rate.cv = mean(five_fold.cv$profit/budget)
fmt(profit_rate.cv)

fold_num = c()
accuracy = c()
profit = c()
cutoff_vals = c(0.25, 0.33, 0.5)
accuracy.cv_list = c()
profit.cv_list = c()
profit_rate.cv_list = c()
loop = 0
combo_num = 0
tune = data.frame()
combos = list(big_growth ~ PC1, big_growth ~ PC2, big_growth ~ PC3, big_growth ~ PC1+PC2, big_growth ~ PC1+PC3, big_growth ~ PC2+PC3, big_growth ~ PC1+PC2+PC3)
variables = c("PC1, big_growth", "PC2, big_growth", "PC3, big_growth", "PC1, PC2, big_growth", "PC1, PC3, big_growth", "PC2, PC3, big_growth", "PC1, PC2, PC3, big_growth")
var_list = c()

for (combo in combos) {
    
    combo_num = combo_num + 1
    
    for (cutoff in cutoff_vals) {

        loop = loop + 1

        for (iteration in 1:5) {
            fold_num[iteration] = iteration
            data.test  = data[,2:9][fold[[iteration]],]
            data.train = data[,2:9][setdiff(1:nrow(data), fold[[iteration]]),]
            model = svm(combo, data.train, type="C-classification", kernel="polynomial", degree=3, cost=10, scale=TRUE, probability=TRUE)
            prob = attr(predict(model, data.test, probability=TRUE), "probabilities")
            big_growth.predicted = as.class(prob, class="YES", cutoff=0.5)
            data.test$big_growth.predicted = big_growth.predicted
            data.test$prob_yes = prob[,2]
            CM = confusionMatrix(data.test$big_growth.predicted, data.test$big_growth)$table
            cm = CM/sum(CM)
            accuracy[iteration] = cm[1,1]+cm[2,2]


            top_twelve = data.test[order(-data.test$prob_yes),][1:12,]
            growth_allocation = 0
            
            for (i in 1:12) {
                growth_allocation = growth_allocation + (1+top_twelve$growth[i])*allocation[i]
            }
            profit[iteration] = growth_allocation - budget
        }
        accuracy.cv_list = c(accuracy.cv_list, mean(accuracy))
        profit.cv_list = c(profit.cv_list,mean(profit))
        profit_rate.cv_list = c(profit_rate.cv_list,mean(profit/budget))
        var_list = c(var_list, variables[combo_num])
    }
}
tune = rbind(tune, data.frame(method="SVM", variables = var_list, cutoff = cutoff_vals, accuracy.cv = accuracy.cv_list, profit.cv=profit.cv_list, profit_rate.cv=profit_rate.cv_list))

# Retrieve "Investment Opportunities.csv"
# Present the dataset size ...

datax = read.csv("Investment Opportunities.csv", header=TRUE)
size(datax)

# Transform representation of the investment opportunity data as required to match the
# representation of the orginal ORIGINAL data.
datax$quarter = quarter(mdy(datax[,2]))

data.current.q1 = datax[(datax$quarter==1) & !is.na(datax$prccq), -ncol(datax)]
data.current.q2 = datax[(datax$quarter==2) & !is.na(datax$prccq), -ncol(datax)]
data.current.q3 = datax[(datax$quarter==3) & !is.na(datax$prccq), -ncol(datax)]
data.current.q4 = datax[(datax$quarter==4) & !is.na(datax$prccq), -ncol(datax)]

data.current.q1 = data.current.q1[!duplicated(data.current.q1$gvkey),]
data.current.q2 = data.current.q2[!duplicated(data.current.q2$gvkey),]
data.current.q3 = data.current.q3[!duplicated(data.current.q3$gvkey),]
data.current.q4 = data.current.q4[!duplicated(data.current.q4$gvkey),]

colnames(data.current.q1)[-c(1, 10, 12)] = paste0(colnames(data.current.q1)[-c(1, 10, 12)], ".q1")
colnames(data.current.q2)[-c(1, 10, 12)] = paste0(colnames(data.current.q2)[-c(1, 10, 12)], ".q2")
colnames(data.current.q3)[-c(1, 10, 12)] = paste0(colnames(data.current.q3)[-c(1, 10, 12)], ".q3")
colnames(data.current.q4)[-c(1, 10, 12)] = paste0(colnames(data.current.q4)[-c(1, 10, 12)], ".q4")
layout(fmt(size(data.current.q1)),
       fmt(size(data.current.q2)),
       fmt(size(data.current.q3)),
       fmt(size(data.current.q4)))
m12 = merge(data.current.q1, data.current.q2, by=c("gvkey", "tic", "conm"), all=TRUE)
m34 = merge(data.current.q3, data.current.q4, by=c("gvkey", "tic", "conm"), all=TRUE)
data.current = merge(m12, m34, by=c("gvkey", "tic", "conm"), all=TRUE, sort=TRUE)

data.current = data.current[!is.na(data.current$prccq.q4),]

size(data.current)

filtered_vars = readRDS('My Filter.rds')
data.current = data.current[, filtered_vars]
size(data.current)

imputed_vals = readRDS('My Imputation.rds')
data.imputed = put_impute(data.current, imputed_vals)
size(data.imputed)

pc = readRDS('My PC.rds')
data.pc = predict(pc, data.imputed)
size(data.pc)

pred_vars = readRDS('My Predictors.rds')
data.pred = data.imputed[,pred_vars[1:3]]
data.pc.3 = data.pc[,1:3]
invest_opp = cbind(data.pred, data.pc.3)
size(invest_opp)
invest_opp

# Stacking with 3 SVM and then using Naive Bayes

prob.1 = attr(predict(model.1, invest_opp, probability=TRUE), "probabilities")
class.predicted.1 = as.class(prob.1, "YES", cutoff=0.25)

prob.2 = attr(predict(model.2, invest_opp, probability=TRUE), "probabilities")
class.predicted.2 = as.class(prob.2, "YES", cutoff=0.25)

prob.3 = attr(predict(model.3, invest_opp, probability=TRUE), "probabilities")
class.predicted.3 = as.class(prob.3, "YES", cutoff=0.25)

data.stack.2 = data.frame(invest_opp, class.predicted.1, class.predicted.2, class.predicted.3)
head(data.stack.2)

data.stack = data.stack.1[,c('big_growth','class.predicted.1','class.predicted.2','class.predicted.3')]
model.stack = naiveBayes(big_growth~class.predicted.1+class.predicted.2+class.predicted.3, data.stack)
output.predicted = predict(model.stack, data.stack.2[,c('class.predicted.1','class.predicted.2','class.predicted.3')], type="raw")
prob.stack = data.frame("YES"=output.predicted, "NO"=1-output.predicted)[,c(1:2)]
names(prob.stack) = c("YES","NO")
invest_opp$prob_yes = prob.stack[,1]
top_invest =  invest_opp[order(invest_opp$prob_yes, decreasing=TRUE),]
portfolio = top_invest[1:12,1:3]
portfolio$allocation = allocation
fmt(portfolio,"portfolio")

# Use the model to predict growths of each investment opportunity.
# Recommend a portfolio of allocations to 12 investment opportunities: gvkey, tic, conm, allocation
# FINAL RECOMMENDATION

# Boosting with SVM
prob = attr(predict(model.2, invest_opp, probability=TRUE), "probabilities")
invest_opp$prob_yes = prob[,1]
top_invest = invest_opp[order(invest_opp$prob_yes, decreasing=TRUE),]
portfolio = top_invest[1:12,1:3]
portfolio$allocation = allocation
fmt(portfolio, "portfolio")
final_portfolio = portfolio

# Neural Network

output = compute(model, invest_opp, rep=1)$net.result
prob = data.frame("YES"=output, "NO"=1-output)
invest_opp$prob_yes = prob[,1]
top_invest =  invest_opp[order(invest_opp$prob_yes, decreasing=TRUE),]
portfolio = top_invest[1:12,1:3]
portfolio$allocation = allocation
fmt(portfolio,"portfolio")

write.csv(final_portfolio, paste0(analyst, ".csv"), row.names=FALSE)


portfolio.retrieved = read.csv(paste0(analyst, ".csv"), header=TRUE)
opportunities = unique(read.csv("Investment Opportunities.csv", header=TRUE)$gvkey)

columns = all(colnames(portfolio.retrieved) == c("gvkey", "tic", "conm", "allocation"))
companies = all(portfolio.retrieved$gvkey %in% opportunities)
allocations = round(sum(portfolio.retrieved$allocation)) == budget
                         
check = data.frame(analyst, columns, companies, allocations)
fmt(check, "Portfolio Recommendation | Format Check")
