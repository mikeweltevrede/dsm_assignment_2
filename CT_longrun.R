#####################################################################
#											RANDOM FORESTS 															  #
#####################################################################
rm(list=ls())

# install.packages(c("psych", "graphics", "sandwich", "bbmle", "pROC",
#                    "randomForest", "xtable"))

library(psych)
library(graphics)
library(sandwich)
library(bbmle)
library(pROC)
library(randomForest)
library(xtable)

###############################################################
#												PREPARATION													  #
###############################################################

data_path = "data"
df_data  = read.table(paste0(data_path, "/R_class.csv"), sep=",", dec=".",
                      header=TRUE)

ca = grep("ca", names(df_data), value=T)
drops = names(df_data) %in% c(ca)
df_data = df_data[!drops]

# drop vars not used
assets = grep("assets", names(df_data), value=T)
stocks = grep("stocks", names(df_data), value=T)
narrowm = grep("narrowm", names(df_data), value=T)
money = grep("money", names(df_data), value=T)
ltrate = grep("ltrate", names(df_data), value=T)
stir = grep("stir", names(df_data), value=T)
loans = grep("loans", names(df_data), value=T)
debt = grep("debt", names(df_data), value=T)
er = grep("er", names(df_data), value=T)
cpi = grep("cpi", names(df_data), value=T)
gap = grep("gap", names(df_data), value=T)
glo = grep("a_", names(df_data), value=T)
gdp = grep("gdp", names(df_data), value=T)
i = grep("i_", names(df_data), value=T)
c = grep("c_", names(df_data), value=T)
ri = grep("ri", names(df_data), value=T)
rc = grep("rc", names(df_data), value=T)

drops = names(df_data) %in% c("year", "ccode", stocks, money, stir, assets, i,
                             ri, glo) # true-false indicator: true at the names in vector
saves = names(df_data) %in% c(glo)
full = df_data[!drops] # drops those variables which have true indication in "drops"
full = cbind(df_data[glo], full)

# FULL SET: omit observations with missing values
full_om = na.omit(full)
sum(full_om$b2)/2

# SELECTION SET:
sel.list = c("b2", "loans1_y_gap", "pdebt_gap", "narrowm_y_gap",  "rltrate",
              "gr_rgdp", "gr_cpi",  "er_gap", "loans1_y", "pdebt", "ltrate")
location = names(full) %in% c(sel.list) # get location of independent var
name.sel = names(full[location]) # get names of features
sel = full[name.sel]
sel_om = na.omit(sel)
sum(sel_om$b2)/2


#DATA for logit model
## interaction-terms for logit model
ia_pub=df_data$pdebt_gap*df_data$ltrate
df_data$ia_pub=ia_pub

ia_prb=df_data$loans1_y_gap*df_data$ltrate
df_data$ia_prb=ia_prb

ia_jb=df_data$loans1_y_gap*df_data$ltrate*df_data$pdebt_gap
df_data$ia_jb=ia_jb

ia_lygr=df_data$loans1_y*df_data$gr_rgdp
df_data$ia_lygr=ia_lygr

ia_pygr=df_data$pdebt*df_data$gr_rgdp
df_data$ia_pygr=ia_pygr

ia_lyer=df_data$loans1_y_gap*df_data$er_gap
df_data$ia_lyer=ia_lyer

## country factor
df_data$country.factor=as.factor(df_data$ccode)

#throw out vars not used
drops.logit = names(df_data) %in% c("year") # true-false indicator: true at the names in vector
full.logit = df_data[!drops] # drops those variables which have true indication in "drops"

#number of trees
trees=5000

###############################################################
#														ANALYSIS   												#
###############################################################

### CLASSIFICATION-TREE ANALYSIS
################################################################################
# variables
var.list = c( "loans1_y_gap", "pdebt_gap", "narrowm_y_gap",  "rltrate",
               "gr_rgdp", "gr_cpi",  "er_gap", "loans1_y", "pdebt", "ltrate")
# model list
model.list = c("Single Tree", "Bagging", "Random Forest")
model.list2 = c("\\textbf{Parameter}","Single", "Bagging", "RF","Single",
                 "Bagging", "RF" )

# variables (logit)
var.logit = c("loans1_y_gap", "pdebt_gap", "narrowm_y_gap",  "rltrate",
               "gr_rgdp", "gr_cpi",  "er_gap")
# interaction terms (logit)
ia.logit = c("ia_pub", "ia_prb", "ia_jb", "ia_lygr", "ia_pygr", "ia_lyer")


# parameter list
param.list = c("B", "$ J_{try} $", "$ J $", "\\# of crises")
out.list = c("\\textbf{Model}", "AUC", "95\\%-CI", "N", "", "AUC", "95\\%-CI",
              "N")

# miscellaneous non-independent
misc.list = c("b2","b1","b3","rec1","rec2","rec3")

# table matrices
out = matrix(nrow=3, ncol=9)
spec = matrix(nrow=4, ncol=7)
sig_base = matrix(nrow=3,ncol=2)
sig_pre = matrix(nrow=2,ncol=2)
sig_many = matrix(nrow=3,ncol=1)

# Nodesizes
node.size = matrix(seq(5,100,5))

# Bootstrap runs
runs = 100

# confidence intervals
n.ci = 3
ci = c(0.99, 0.95, 0.9)
################################################################################


#LOGIT
aucs = matrix(nrow=1, ncol=runs)
ci95_lo = matrix(nrow=1, ncol=runs)
ci95_up = matrix(nrow=1, ncol=runs)

N = matrix(nrow=1, ncol=runs)
	

# get formula
location = names(full.logit) %in% c(var.logit, ia.logit,"country.factor") # get location of vars
name = names(full.logit[location]) # get names
indep = paste(name, collapse="+") # indep. variables
dep = paste("b2~") # dep. variable
fmla = as.formula(paste(dep, indep)) # get formula


for(j in 1:runs) {
	
	# training, test sample
	set.seed(j)
	indexes = sample(1:nrow(full.logit), size=0.632*nrow(full), replace=F)
	test = full.logit[-indexes,]
	train = full.logit[indexes,]
	
	# Regression
	logit=glm(fmla, data=train, family="binomial")
	N[1,j] = logit$df.null

	# OOS-analysis
	pred=predict(logit, newdata=test, type="response") # predicted outcome

	location = names(test) %in% c("b2")
	name = names(test[location]) # get names
	true=test[,name] # real outcome

	r=roc(true,pred,ci=T) # ROC analysis
	aucs[1,j] = as.numeric(r$auc)
		
	ci95_lo[1,j] = as.numeric(ci.auc(r,conf.level=ci[2]))[1]
	ci95_up[1,j] = as.numeric(ci.auc(r,conf.level=ci[2]))[3]

}

N = as.numeric(colMeans(as.matrix(N[1, ]))) # update output table matrix

auc=as.numeric(colMeans(as.matrix(aucs[1, ])))
ci95_lo=as.numeric(colMeans(as.matrix(ci95_lo[1, ])))
ci95_up=as.numeric(colMeans(as.matrix(ci95_up[1, ])))



# Representative logit model whose AUC equals the MCCV average

# training, test sample
set.seed(4)
indexes = sample(1:nrow(full.logit), size=0.632*nrow(full.logit), replace=F)
test = full.logit[-indexes,]
train = full.logit[indexes,]
	
# Regression
logit=glm(fmla, data=train, family="binomial")

# OOS-analysis
pred=predict(logit, newdata=test, type="response") # predicted outcome

true=test[,"b2"] # real outcome

library(pROC)
r_log=roc(true,pred,ci=F) # ROC analysis
r_log



## SINGLE TREE-selection
library(randomForest)

location = names(sel_om) %in% c(var.list) # get location of dependent var
name.indep = names(sel_om[location]) # get names of features
indep = sel_om[name.indep]
location = names(sel_om) %in% c("b2") # get location of dependent var
name.dep = names(sel_om[location])
dep = factor(sel_om[,"b2"]>0) # dep. var.

# Define matrices
aucs = matrix(nrow=1, ncol=runs)
ci95_lo = matrix(nrow=1, ncol=runs)
ci95_up = matrix(nrow=1, ncol=runs)

for(j in 1:runs){
	set.seed(j)
	tree_selection = randomForest(indep, y=dep,
	 data=sel_om,
	 ntree=1,
	 replace=T, # bootstrapping (with replacement!)
	 mtry=(ncol(indep)), # all features except dependent variable
 
	 cutoff=c(1/2, 1/2), # majority vote: class with maximum ratio of (prop. of votes/cutoff(=1/k)) wins
	 sampsize=nrow(sel_om), # bootstrapping (comput. more efficient wihtout much loss by using 1/2*train (see Friedman & Hall, 	2007))
	 nodesize=10 # fully grow trees (to avoid overfitting (see Segal, 2004)); (also see Biau et al., 2012 on consistency)
	 ) 
	tree_selection

	# OOS-analysis
	library(pROC)
  
	# predicted outcome; second column = TRUE probability (votes combined with 
	# normvotes=T equals type="prob")
	pred = predict(tree_selection, type="prob")[,2]

	true = sel_om[,name.dep]

	r=roc(true, pred, ci=T) # ROC analysis
	aucs[1,j] = as.numeric(r$auc)		
	ci95_lo[1,j] = as.numeric(ci.auc(r,conf.level=ci[2]))[1]
	ci95_up[1,j] = as.numeric(ci.auc(r,conf.level=ci[2]))[3]
}

out[1,1]=model.list[1]
out[1,2]=as.numeric(colMeans(as.matrix(aucs[1, ])))
out[1,3]=as.numeric(colMeans(as.matrix(ci95_lo[1, ])))
out[1,4]=as.numeric(colMeans(as.matrix(ci95_up[1, ])))
out[1,5]=nrow(sel_om)

	
spec[1,2]=tree_selection$ntree
spec[2,2]=tree_selection$mtry


# Representative tree whose AUC equals the MCCV average
set.seed(4)
tree_selection= randomForest(indep, y=dep,
 data=sel_om,
 ntree=1,
 replace=T, # bootstrapping (with replacement!)
 mtry=(ncol(indep)), # all features except dependent variable
 
 cutoff=c(1/2, 1/2), # majority vote: class with maximum ratio of (prop. of votes/cutoff(=1/k)) wins
 sampsize=nrow(sel_om), # bootstrapping (comput. more efficient wihtout much loss by using 1/2*train (see Friedman & Hall, 	2007))
 nodesize=10 # fully grow trees (to avoid overfitting (see Segal, 2004)); (also see Biau et al., 2012 on consistency)
 ) 
tree_selection

# OOS-analysis
library(pROC)

# predicted outcome; second column = TRUE probability (votes combined with
# normvotes=T equals type="prob")
pred = predict(tree_selection, type="prob")[,2]

true = sel_om[,name.dep]

r_tree=roc(true, pred, ci=T) # ROC analysis
r_tree


# compare ROCs
testobj = roc.test(r_tree,r_log,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_base[1,1]=testobj$p.value[1]




## BAGGING-selection
library(randomForest)

location = names(sel_om) %in% c(var.list) # get location of independent var
name.indep = names(sel_om[location]) # get names of features
location = names(sel_om) %in% c("b2") # get location of dependent var
name.dep = names(sel_om[location]) # get name of dep. var.
indep = sel_om[name.indep]
dep = factor(sel_om[name.dep]>0)

# grow trees
set.seed(1)
bagging_selection= randomForest(indep, y=dep,
 data=sel_om,
 ntree=trees,
 replace=T, # bootstrapping (with replacement!)
 mtry=(ncol(indep)), # all features except dependent variable
 
 cutoff=c(1/2, 1/2), # majority vote: class with maximum ratio of (prop. of votes/cutoff(=1/k)) wins
 sampsize=nrow(sel_om), # bootstrapping (comput. more efficient wihtout much loss by using 1/2*train (see Friedman & Hall, 2007))
 nodesize=1 # fully grow trees (experiment to avoid overfitting (see Segal, 2004)); (also see Biau et al., 2012 on consistency)
 ) 
bagging_selection

# convergence diagnostic
palette("default")
plot(bagging_selection, type="l", main="")

# OOS-analysis
library(pROC)

# predicted outcome; second column = TRUE probability (votes combined with
# normvotes=T equals type="prob")
pred = predict(bagging_selection, type="prob")[,2]

true = sel_om[,name.dep]

r=roc(true, pred, ci=T) # ROC analysis

out[2,1]=model.list[2]
out[2,2] = as.numeric(r$auc)
out[2,3]=as.numeric(ci.auc(r,conf.level=0.95))[1]
out[2,4]=as.numeric(ci.auc(r,conf.level=0.95))[3]
out[2,5]=nrow(sel_om)

spec[1,3]=bagging_selection$ntree
spec[2,3]=bagging_selection$mtry
spec[3,3]=ncol(indep)
spec[4,3]=floor(sum(sel_om$b2)/2)

# compare ROCs
r_bag=r
testobj = roc.test(r_bag,r_log,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_base[2,1]=testobj$p.value[1]

testobj = roc.test(r_bag,r_tree,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_pre[1,1]=testobj$p.value[1]



## RF-selection
library(randomForest)

location = names(sel_om) %in% c(var.list) # get location of independent var
name.indep = names(sel_om[location]) # get names of features
location = names(sel_om) %in% c("b2") # get location of dependent var
name.dep = names(sel_om[location]) # get name of dep. var.
indep = sel_om[name.indep]
dep = factor(sel_om[name.dep]>0)

# grow trees
set.seed(1)
rf_selection= randomForest(indep, y=dep,
 data=sel_om,
 ntree=trees,
 replace=T, # bootstrapping (with replacement!)
 mtry=sqrt(ncol(indep)), # all features except dependent variable
 importance=T,
 cutoff=c(1/2, 1/2), # majority vote: class with maximum ratio of (prop. of votes/cutoff(=1/k)) wins
 sampsize=nrow(sel_om), # bootstrapping (comput. more efficient wihtout much loss by using 1/2*train (see Friedman & Hall, 2007))
 nodesize=1 # fully grow trees (experiment to avoid overfitting (see Segal, 2004)); (also see Biau et al., 2012 on consistency)
 ) 
rf_selection

perm1= importance(rf_selection,type=1,scale=F) # don't scale (see Strobl & Zeilleis)
ord1=order(perm1)
names1=rownames(perm1)[ord1]

purity= importance(rf_selection,type=2)
ord2=order(purity)
names2=rownames(purity)[ord2]

pdf('/Users/felixward/Dropbox/CrisisPrediction/Written/Importance1.pdf', width=12, height=4)

op=par(
mfrow=c(1,2),
oma=c(0.5,0, 0, 0),
mar=c(1, 12, 1, 1),
mgp=c(1.5, 0.5, 0)) #axis.title.position, axis.label.position, axis.line.position

barplot(as.vector(perm1[ord1]), main="Permutation Importance", ylab="", las=2,
        xlab="", las=1, axisnames=T,
        names.arg=rev(c("CPI (gr)", "Loans/GDP (gap)", "LT Rate (r)", 
                        "LT Rate (n)", "Loans/GDP", "Public Debt/GDP", 
                        "Public Debt/GDP (gap)", "Exchange Rate (gap)", 
                        "Narrow money/GDP (gap)","GDP (r)(gr)")),
col=c("white"),
horiz=T)

barplot(as.vector(purity[ord2]), main="Gini Importance", ylab="", las=2,
        xlab="", las=1, axisnames=T,
        names.arg=c("LT Rate (r)", "CPI (gr)", "Narrow money/GDP (gap)", 
                    "LT Rate", "GDP (r)(gr)", "Public Debt/GDP (gap)", 
                    "Public Debt/GDP", "Exchange Rate (gap)", "Loans/GDP", 
                    "Loans/GDP (gap)"),
col=c("white"),
horiz=T)

dev.off()

	
# convergence diagnostic
palette("default")
plot(rf_selection, type="l", main="")

# OOS-analysis
library(pROC)

# predicted outcome; second column = TRUE probability (votes combined with
# normvotes=T equals type="prob")
pred = predict(rf_selection, type="prob")[,2] 

true = sel_om[,name.dep]

r=roc(true, pred, ci=T) # ROC analysis

out[3,1]=model.list[3]
out[3,2] = as.numeric(r$auc)
out[3,3]=as.numeric(ci.auc(r,conf.level=0.95))[1]
out[3,4]=as.numeric(ci.auc(r,conf.level=0.95))[3]
out[3,5]=nrow(sel_om)

spec[1,4]=rf_selection$ntree
spec[2,4]=rf_selection$mtry

# compare ROCs
r_rf=r
testobj = roc.test(r_rf,r_log,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_base[3,1]=testobj$p.value[1]

testobj = roc.test(r_rf,r_bag,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_pre[2,1]=testobj$p.value[1]


## SINGLE TREE-full
library(randomForest)

location = names(full_om) %in% c(misc.list) # get location of dependent var
name.indep = names(full_om[!location]) # get names of features
indep = full_om[name.indep]
dep = factor(full_om[,"b2"]>0) # dep. var.

# define matrices
aucs = matrix(nrow=1, ncol=runs)
ci95_lo = matrix(nrow=1, ncol=runs)
ci95_up = matrix(nrow=1, ncol=runs)

for(j in 1:runs){
	set.seed(j)
	tree_full= randomForest(indep, y=dep,
	 data=full_om,
	 ntree=1,
	 replace=T, # bootstrapping (with replacement!)
	 mtry=(ncol(indep)), # all features except dependent variable
 
	 cutoff=c(1/2, 1/2), # majority vote: class with maximum ratio of (prop. of votes/cutoff(=1/k)) wins
	 sampsize=nrow(full_om), # bootstrapping (comput. more efficient wihtout much loss by using 1/2*train (see Friedman & Hall, 	2007))
	 nodesize=10 # fully grow trees (experiment to avoid overfitting (see Segal, 2004)); (also see Biau et al., 2012 on consistency)
	 ) 
	tree_full

	# OOS-analysis
	library(pROC)
	
  # predicted outcome; second column = TRUE probability (votes combined with 
	# normvotes=T equals type="prob")
	pred = predict(tree_full, type="prob")[,2]

	true = full_om[,name.dep]

	r=roc(true, pred, ci=T) # ROC analysis
	aucs[1,j] = as.numeric(r$auc)		
	ci95_lo[1,j] = as.numeric(ci.auc(r,conf.level=ci[2]))[1]
	ci95_up[1,j] = as.numeric(ci.auc(r,conf.level=ci[2]))[3]
}

out[1,6]=as.numeric(colMeans(as.matrix(aucs[1, ])))
out[1,7]=as.numeric(colMeans(as.matrix(ci95_lo[1, ])))
out[1,8]=as.numeric(colMeans(as.matrix(ci95_up[1, ])))
out[1,9]=nrow(full_om)
	
spec[1,5]=tree_full$ntree
spec[2,5]=tree_full$mtry


# compare ROCs
r_tree_m=r
testobj = roc.test(r_tree_m,r_log,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_base[1,2]=testobj$p.value[1]

testobj = roc.test(r_tree_m,r_tree,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_many[1,1]=testobj$p.value[1]



## BAGGING-all variables
library(randomForest)

location = names(full_om) %in% c(misc.list) # get location of dependent var
name.indep = names(full_om[!location]) # get names of features
indep = full_om[name.indep]
dep = factor(full_om[,"b2"]>0) # dep. var.

# grow trees
set.seed(1)
bagging_full= randomForest(indep, y=dep,
 data=full_om,
 ntree=trees,
 replace=T, # bootstrapping (with replacement!)
 mtry=ncol(indep), # all features except dependent variable
 
 cutoff=c(1/2, 1/2), # majority vote: class with maximum ratio of (prop. of votes/cutoff(=1/k)) wins
 sampsize=nrow(full_om), # bootstrapping (comput. more efficient wihtout much loss by using 1/2*train (see Friedman & Hall, 2007))
 nodesize=1 # fully grow trees (experiment to avoid overfitting (see Segal, 2004)); (also see Biau et al., 2012 on consistency)
 ) 
bagging_full

# convergence diagnostic
palette("default")
plot(bagging_full, type="l", main="")

# OOS-analysis
library(pROC)

# predicted outcome; second column = TRUE probability (votes combined with
# normvotes=T equals type="prob")
pred = predict(bagging_full, type="prob")[,2]

true = full_om[,name.dep]

r=roc(true, pred, ci=T) # ROC analysis
out[2,6] = as.numeric(r$auc)
out[2,7] = as.numeric(ci.auc(r,conf.level=0.95))[1]
out[2,8] = as.numeric(ci.auc(r,conf.level=0.95))[3]
out[2,9]=nrow(full_om)


spec[1,6]=bagging_full$ntree
spec[2,6]=bagging_full$mtry
spec[3,6]=ncol(indep)
spec[4,6]=floor(sum(full_om$b2)/2)

# compare ROCs
r_bag_m=r
testobj = roc.test(r_bag_m,r_log,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_base[2,2]=testobj$p.value[1]


testobj = roc.test(r_bag_m,r_tree_m,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_pre[1,2]=testobj$p.value[1]


testobj = roc.test(r_bag_m,r_bag,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_many[2,1]=testobj$p.value[1]




## RANDOM FOREST
library(randomForest)

location = names(full_om) %in% c(misc.list) # get location of dependent var
name.indep = names(full_om[!location]) # get names of features
indep = full_om[name.indep]
dep = factor(full_om[,"b2"]>0) # dep. var.

# grow trees
set.seed(1)
rf_full= randomForest(indep, y=dep,
 data=full_om,
 ntree=trees,
 importance=T,
 replace=T, # bootstrapping (with replacement!)
 mtry=sqrt(ncol(indep)), # all features except dependent variable
 
 cutoff=c(1/2, 1/2), # majority vote: class with maximum ratio of (prop. of votes/cutoff(=1/k)) wins
 sampsize=nrow(full_om), # bootstrapping (comput. more efficient wihtout much loss by using 1/2*train (see Friedman & Hall, 2007))
 nodesize=1 # fully grow trees (experiment to avoid overfitting (see Segal, 2004)); (also see Biau et al., 2012 on consistency)
 ) 
rf_full

perm1= importance(rf_full,type=1,scale=F) # don't scale (see Strobl & Zeilleis)
ord1=order(-perm1)
names1=rownames(perm1)[ord1][1:10]

purity= importance(rf_full,type=2)
ord2=order(-purity)
names2=rownames(purity)[ord2][1:10]

pdf('/Users/felixward/Dropbox/CrisisPrediction/Written/Importance2.pdf',
    width=12, height=4)

op=par(
mfrow=c(1,2),
oma=c(0.5,0, 0, 0),
mar=c(1, 12, 1, 1),
mgp=c(1.5, 0.5, 0)) #axis.title.position, axis.label.position, axis.line.position

barplot(rev(as.vector(perm1[ord1][1:10])), main="Permutation Importance",
        ylab="", las=2, xlab="", las=1, axisnames=T,
        names.arg=rev(c("I/GDP (gap)(glo)", "C/GDP (glo)",
                        "ST Rate (n)(gap)(glo)", "I/GDP (glo)", 
                        "ST Rate (gap)(glo)", "I/GDP (gr)(glo)", 
                        "ST Rate (r)(glo)", "Bank Assets (r)(gap)(glo)", 
                        "Public Debt (gap)(glo)", "Loans (r)(gap)")),
col=c("white"),
horiz=T)

barplot(rev(as.vector(purity[ord2][1:10])), main="Gini Importance", ylab="",
        las=2, xlab="", las=1, axisnames=T,
        names.arg=c("Loans (r)(gap)", "Loans (r)(gr)", "Loans/GDP (gap)", 
                    "Loans/GDP", "Exchange Rate", "Loans/GDP (gr)", 
                    "Exchange Rate (gr)", "ST Rate (gap)(glo)", "C/GDP", 
                    "LT Rate (n)"),
col=c("white"),
horiz=T)

dev.off()


# convergence diagnostic
palette("default")
plot(rf_full, type="l", main="")

# OOS-analysis
library(pROC)

# predicted outcome; second column = TRUE probability (votes combined with
# normvotes=T equals type="prob")
pred = predict(rf_full, type="prob")[,2]

true = full_om[,"b2"]

r=roc(true, pred, ci=T) # ROC analysis
out[3,6] = as.numeric(r$auc)
out[3,7] = as.numeric(ci.auc(r,conf.level=0.95))[1]
out[3,8] = as.numeric(ci.auc(r,conf.level=0.95))[3]
out[3,9]=nrow(full_om)

spec[1,7]=rf_full$ntree
spec[2,7]=rf_full$mtry

# compare ROCs
r_rf_m=r
testobj = roc.test(r_rf_m,r_log,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_base[3,2]=testobj$p.value[1]


testobj = roc.test(r_rf_m, r_bag_m,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_pre[2,2]=testobj$p.value[1]


testobj = roc.test(r_rf_m,r_rf,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

sig_many[3,1]=testobj$p.value[1]



out
spec

sig_base
sig_pre
sig_many

save.image("/Users/felixward/Dropbox/CrisisPrediction/DoFiles/CT_longrun") 


###############################################################
#													ROC CURVES													#
###############################################################
load("/Users/felixward/Dropbox/CrisisPrediction//DoFiles/CT_longrun")

## RF
library(pROC)
r_rf2=paste("RF: AUC=",round(r_rf_m$auc[1],2), sep="")

r_log2=paste("Logit: AUC=",round(r_log$auc[1],2), sep="")

# compare ROCs
testobj = roc.test(r_rf_m,r_log,method="delong",alternative="greater")
options("scipen"=10)
options()$scipen

pval1=testobj$p.value[1]
pval=sprintf("%.2f",pval1)


size = 0.6
pdf('/Users/felixward/Dropbox/CrisisPrediction/Written/ROC.pdf', width=2*2.95,
    height=2*2.95)
plot(r_log, lwd=0.5, print.thres=F, print.auc=F,
     print.auc.pattern=sprintf("%.18s", r_log2), auc.polygon=F,
     max.auc.polygon=F, print.auc.x=.5,  print.auc.y=.775, xaxt='n', yaxt='n',
     xlab="FPR", ylab="TPR", mgp=c(2.25,0.25,0), cex.axis=size, cex.lab=1,
     identity.lty=2, print.auc.cex=size, tck=-0.025, mar=c(3.5, 3.5, 3.5, 3.5))
 axis(1,at=c(1.0, 0.8, 0.6, 0.4, 0.2, 0), labels=c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
      pos=-0.04)
 axis(2,at=c(1.0, 0.8, 0.6, 0.4, 0.2, 0), pos=1.04)
dev.off()

size =0.75

pdf('/Users/felixward/Dropbox/CrisisPrediction/Written/ROC2.pdf', width=4,
    height=4)

 plot(r, lwd=0.5, print.thres=F, print.auc=T,
      print.auc.pattern=sprintf("%.12s", r_rf2), auc.polygon=F,
      max.auc.polygon=F, print.auc.x=.95, print.auc.y=.95, xlab="1-FPR", 
      ylab="TPR", mgp=c(1.25, 0.5, 0), cex.lab=size, cex.axis=size,
      legacy.axes=F, identity.lty=2, print.auc.cex= size, , tck=-0.01, 
      mar=c(2.5, 2.5, 2, 2))

 par(new=T)
 
plot(r_log, lwd=0.5, print.thres=F, print.auc=T,
     print.auc.pattern=sprintf("%.18s", r_log2), auc.polygon=F,
     max.auc.polygon=F, print.auc.x=.735,  print.auc.y=.825, xaxt='n',
     yaxt='n', ann=F, xlab="", ylab="", mgp=c(0.85,0.25,0),
     identity.lty=2, col="grey50", print.auc.cex= size, mar=c(2.5, 2.5, 2, 2))

par(new=T)


text(0.4,0.025, labels=paste("Logit AUC = RF AUC (p-value) : ", pval), cex=size,
     xaxt='n', yaxt='n', ann=F, xlab="", ylab="", mgp=c(0.85,0.25,0))

dev.off()



###############################################################
#															TABLES													#
###############################################################
library(xtable)

#SPECIFICATION TABLE (always use double the amount of backslashes needed in
# latex)
spec[,1] = param.list

spec2=rbind(model.list2,spec) #get model headers
# get rid of row and columnnames
x = data.frame(spec2)
spec2=as.matrix(x)
rownames(spec2) = rep("", nrow(spec2))
colnames(spec2) = rep("", ncol(spec2))


#OUTPUT TABLE (always use double the amount of backslashes needed in latex)

# reformat decimals
out2=out
out2[,2:9]=round(as.numeric(out2[,2:9]), digits=2)

# add symbols for significance

for (i in 1:nrow(sig_base)){
	if(sig_base[i,1]<=0.05) {
		out2[i,2] = paste("\\textbf{",out2[i,2],"}",collapse="")
	}
	if(sig_base[i,2]<=0.05) {
		out2[i,6] = paste("\\textbf{",out2[i,6],"}",collapse="")
	}	
}

# # for (i in 1:nrow(sig_pre)){
	# if(sig_pre[i,1]<=0.05) {
		# out2[i+1,2] = paste(out2[i+1,2],"$^{\\ddagger}$",collapse="")
	# }
	# if(sig_pre[i,2]<=0.05) {
		# out2[i+1,6] = paste(out2[i+1,6],"$^{\\ddagger}$",collapse="")
	# }	
# }

for (i in 1:nrow(sig_many)){
	if(sig_many[i,1]<=0.05) {
		out2[i,6] = paste(out2[i,6],"$^{\\mathsection}$",collapse="")
	}	
}


# confidence intervals
cis=paste(out2[,3], out2[,4], sep=",")
cis=paste("[", cis, sep="")
cis=paste(cis, "]", sep="")

cis2=paste(out2[,7], out2[,8], sep=",")
cis2=paste("[", cis2, sep="")
cis2=paste(cis2, "]", sep="")

out3 = out2[,c(1,2,5,6,9)] # leave out .9, .99 lower-ci columns
out4 = cbind(out3,cis,cis2)
out5 = out4[,c(1,2,6,3,4,7,5)] 
out6 = out5[,1:4]
out7 = cbind(out6,matrix(nrow=3, ncol=1)) # insert empy column
out8 = cbind(out7,out5[,5:7])
outF=rbind(out.list,out8) #get model headers
# get rid of row and columnnames
x = data.frame(outF)
outF=as.matrix(x)
rownames(outF) = rep("", nrow(outF))
colnames(outF) = rep("", ncol(outF))

spec3 = spec2[,1:4]
spec4 = cbind(spec3,matrix(nrow=5, ncol=1))
specF = cbind(spec4,spec2[,5:7])

#COMBINED
comb=rbind(outF,specF)

mat3=xtable(comb, align="llccccccc", caption="CT-EWS",
             label="tab:CT_out") # for whatever reason need one column more than I actually want (added "l" to left)

print(mat3, type="latex", caption.placement="top", hline.after=c(-1,nrow(mat3)), 
      sanitize.text.function = function(x){x}, file="CT.txt", replace=T,
      floating=F, booktabs=T, include.colnames=F, include.rownames=F, 
      add.to.row=list(pos=list(0,0,0,0,1,4,4,4,4,5), 
                      command=c(" \\multicolumn{1}{c}{} & \\multicolumn{7}{c}{\\textbf{Results}} \\\\",
                                "  \\cmidrule(l r){2-8} \\\\",
                                " \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{\\textbf{Restricted Selection}} & \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{\\textbf{Many Predictors}} \\\\",
                                "  \\cmidrule(l r){2-4} \\cmidrule(l r){6-8} \\\\",
                                " \\cdashline{1-8} \\\\",
                                " \\multicolumn{1}{c}{} & \\multicolumn{7}{c}{\\textbf{Specification}} \\\\",
                                "  \\cmidrule(l r){2-8} \\\\",
                                " \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{\\textbf{Restricted Selection}} & \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{\\textbf{Many Predictors}} \\\\",
                                "  \\cmidrule(l r){2-4}  \\cmidrule(l r){6-8} \\\\",
                                "  \\cdashline{1-8} \\\\")))
