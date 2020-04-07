library(foreign)

data<-read.arff("/Users/pilarpineiro/Google\ Drive/3ro/Segundo\ Cuatri/Mineria\ Datos/R/TrabajoFinal/DLBCL-Stanford/DLBCL-Stanford.arff")
summary(data$Class)
data[is.na(data)] <- 0

library(FSelector)

FilterFeatures<-function(model,data,funct){
  weights<-funct(model,data)
  #for (z in 1:7){ #pruebo con varios cutoff
  subset<-cutoff.k(weights,4)
  f<-as.simple.formula(subset,"Class")
  return(f)
}

filtvbles<-FilterFeatures(Class~.,data,chi.squared)
filtvbless<-FilterFeatures(Class~.,data,information.gain)
library(pROC)
library(hier.part)
library(nnet)
library(FSelector)
library(caret)
library(rminer)

#Holdout:

HoldOutRN<-function (datos,esq.val,s,m,d){
  set.seed(1200)
  obtsamples<-holdout(datos$Class,ratio=esq.val,internalsplit=TRUE,mode="stratified",iter=1,seed=NULL,window=10,increment=1)
  trdata<-datos[obtsamples$tr,] #tomo datos para training
  testdata<-datos[obtsamples$ts,] #tomo datos para test
  
  RN.fit<-nnet(filtvbles, data=trdata,size=s,maxit=m,decay=d)#estimo modelo
  
  vpredrn1<-predict(RN.fit, trdata,type="raw")#prediccion sobre training data
  
  res1obj.rn<-auc(trdata$Class,vpredrn1)
  

  vpredrn2<-predict(RN.fit, testdata,type="raw")#prediccion sobre test data
  res2obj.rn<-auc(testdata$Class,vpredrn2)
  
  result<-c(res1obj.rn,res2obj.rn); #ambos resultados a un vector 1-training 2-test
  
  return(result)
}


RHoldOutRN<-function(datos,n,esq.val,s,m,d){
  replicate(n,HoldOutRN(datos,esq.val,s,m,d))
}


library(pROC)
library(rminer)
library(nnet)
resrn1<-RHoldOutRN(data,30,2/3,5,1100,5e-4)


resrn2<-RHoldOutRN(data,30,2/3,7,1200,5e-2)


resrn3<-RHoldOutRN(data,30,2/3,5,20,5e-1)

mean(resrn1[2,]) 
mean(resrn2[2,])
mean(resrn3[2,])

par(mfrow=c(2,2))
boxplot(cbind(unlist(resrn1[2,])),xlab="RHoldout RN",ylim=c(0.4,1.5))

boxplot(cbind(unlist(resrn2[2,])),xlab="RHoldout RN",ylim=c(0.4,1.5))

boxplot(cbind(unlist(resrn3[2,])),xlab="RHoldout RN",ylim=c(0.4,1.5))





#----------------------------------------
library(caret)
library(rminer)
library(nnet)

ValCrRN<-function (datos,s,m,d){
  set.seed(1200)
  
  folds <- createFolds(datos$Class, k=10)
  restraining <- list();
  restest <- list();
  for(i in 1:10){
    testData <- datos[unlist(folds[i]), ]
    trainData <- datos[-unlist(folds[i]), ]
    
    RN.fit<-nnet(filtvbles, data=trainData,size=s,maxit=m,decay=d)#estimo modelo
    
    vpredrn1<-predict(RN.fit, trainData,type="raw")#prediccion sobre training data
    
    res1obj.rn<-auc(trainData$Class,vpredrn1)
  
    vpredrn2<-predict(RN.fit, testData,type="raw")#prediccion sobre test data
    res2obj.rn<-auc(testData$Class,vpredrn2)
    
    restraining<-c(restraining,res1obj.rn)
    restest<-c(restest,res2obj.rn)
  }
  result<-matrix(ncol=length(restest),nrow=2)
  
  result[1,]<-unlist(restraining); #ambos resultados a un vector 1-training 2-test
  result[2,]<-unlist(restest);
  return(result)
}




rescrval<-ValCrRN(data,5,1100,5e-4)
resrn<-RHoldOutRN(data,30,2/3,5,1100,5e-4)
mean(rescrval[1])

mean(rescrval[2])

mean(resrn[1,])

mean(resrn[2,])

par(mfrow=c(1,2))
boxplot(cbind(unlist(rescrval[2,]),unlist(rescrval[1,])),xlab="CrossVal RN",ylim=c(0.4,1.5))
boxplot(cbind(unlist(resrn[2,]),unlist(resrn[1,])),xlab="RHoldout RN",ylim=c(0.4,1.5))

#Valores atípicos puntos
#---------------------------------------
  #VALCR REGR LOGISTICA



ValCrRL<-function (data,form){
  
  folds <- createFolds(data$Class, k=10)
  restest <- list();
  for(i in 1:10){
    testData <- data[unlist(folds[i]), ]
    trainData <- data[-unlist(folds[i]), ]
    
    
    rl.fit<-glm(form, data=trainData,family=binomial("logit")) # modelo con subset
    res<-ComputeAUCLogRegr (rl.fit,testData) #calculo auc del modelo
    restest<-c(restest,res) #meto auc en lista 
    
  }
  result<-matrix(ncol=length(restest),nrow=1)
  result[1,]<-unlist(restest);
  return(result)
}




ComputeAUCLogRegr <- function(fittedmodel,testData) {
  vpred1<-predict(fittedmodel, testData,type="response")
  
  obj.rl1<-auc(testData$recid,vpred1)
  
  return (obj.rl1)
}






#--------------------------------------
library(e1071)
library(caret)
library(pROC)
library(FSelector)






ValCrSVM<-function (datos,c,g,p){
  
  folds <- createFolds(datos$Class, k=10)
  restraining <- list();
  restest <- list();
  for(i in 1:10){
    testData <- datos[unlist(folds[i]), ]
    trainData <- datos[-unlist(folds[i]), ]
    
    svm.fit <- svm(filtvbles, data=datos, cost = c, gamma =g, probability=p)
    
    pred1 <- predict(svm.fit,trainData,probability=p)
    
    svm.pred1 <- attr(pred1, which="probabilities")[,"germinal"]
    
    svmrestr<-auc(cases=as.numeric(trainData$Class),control=as.numeric(svm.pred1))
    
    pred2 <- predict(svm.fit,testData,probability=p)
    
    svm.pred2 <- attr(pred2, which="probabilities")[,"germinal"]
    
    svmrestst<-auc(cases=as.numeric(testData$Class),control=as.numeric(svm.pred2))
    
    restraining<-c(restraining,svmrestr)
    restest<-c(restest,svmrestst)
  }
  result<-matrix(ncol=length(restest),nrow=2)
  
  result[1,]<-unlist(restraining); #ambos resultados a un vector 1-training 2-test
  result[2,]<-unlist(restest);
  
  return(result)
}

res1C<-ValCrSVM(data,1000,0.01,TRUE)
res2C<-ValCrSVM(data,1900,0.7,TRUE)



mean(res1C[2,])
mean(res2C[2,])

par(mfrow=c(1,2))
boxplot(cbind(unlist(res1C[2],)),xlab="1000,0.01",ylim=c(0.4,1.5))
boxplot(cbind(unlist(res1C[2,])),xlab="1900,0.7",ylim=c(0.4,1.5))



#------------------------------
  
library(caret)
library(rpart)
library(pROC)



ValCrDT<-function (datos,cp){
  folds <- createFolds(datos$Class, k=10)
  restraining <- list();
  restest <- list();
  for(i in 1:10){
    testData <- datos[unlist(folds[i]), ]
    trainData <- datos[-unlist(folds[i]), ]

    
    
    Dt.fit<-rpart(filtvbles, data=trainData,control=rpart.control(cp=cp))
    
    Dt.pred1<-predict(Dt.fit,trainData,type="prob") 
    
    res1obj.dt<-auc(trainData$Class,Dt.pred1[,"germinal"])
    
  
    Dt.pred2<-predict(Dt.fit,testData,type="prob") 
    
    res2obj.dt<-auc(testData$Class,Dt.pred2[,"germinal"])
    
    restraining<-c(restraining,res1obj.dt)
    restest<-c(restest,res2obj.dt)
  }
  result<-matrix(ncol=length(restest),nrow=2)
  
  result[1,]<-unlist(restraining); #ambos resultados a un vector 1-training 2-test
  result[2,]<-unlist(restest);
  return(result)
}


modeloDT<-ValCrDT(data,0.06)
modeloDT2<-ValCrDT(data,0.01)

mean(modeloDT[1,])
mean(modeloDT[2,]) 
mean(modeloDT2[1,])
mean(modeloDT2[2,]) 

  
par(mfrow=c(1,2))
boxplot(cbind(unlist(modeloDT[1,]),unlist(modeloDT[2,])),xlab="0.06",ylim=c(0.4,1.5))
boxplot(cbind(unlist(modeloDT2[1,]),unlist(modeloDT2[2,])),xlab="0.01",ylim=c(0.4,1.5))




#-----------------------------------------

#Vamos a ver como varia distribucion normal representando en histogramas los valores medios obtenidos tras cambiar el valor de n para un solo modelo con estratificado = TRUE:

RHoldOut1<-function(datos,n,esq.val,s,m,d){
  replicate(30,HoldOutRN(datos,esq.val,s,m,d))
}
RHoldOut2<-function(datos,n,esq.val,s,m,d){
  replicate(50,HoldOutRN(datos,esq.val,s,m,d))
}
RHoldOut3<-function(datos,n,esq.val,s,m,d){
  replicate(200,HoldOutRN(datos,esq.val,s,m,d))
}


res1<-RHoldOut1(data,30,2/3,5,100,4e-1)


res2<-RHoldOut2(data,30,2/3,5,300,5e-1)


res3<-RHoldOut3(data,30,2/3,5,300,5e-1)

par(mfrow=c(2,2)) 

hist(as.numeric(res1),xlab="n=30",ylim=c(0,200))
hist(as.numeric(res2),xlab="n=50",ylim=c(0,200))
hist(as.numeric(res3),xlab="n=200",ylim=c(0,200))

#-------------------------------------

#sobre la division de hldout 5 veces y se promedia el resultado del auc. fijar semilla es ara reproducir. Para
#hacer indep. ¿Si hacemos 150 vces el rep holdout y lo comparamos con holdout repitiendolo 30veces en rep holdout pero el holdout con 5 veces el auc y su media?? sale igual?--> si sale igual . con uno y con otro.dos de las medias son: resCMedia
# [1] 0.8588305
# > resSinMedia
# [1] 0.8588305
# > HACER BOXPLOTS! sobre el test media parecida. 



HoldOutRN_5AUC<-function (datos,esq.val,s,m,d){
  set.seed(1200)
  listauc1<-list()
  listauc2<-list()
  obtsamples<-holdout(datos$Class,ratio=esq.val,internalsplit=TRUE,mode="stratified",iter=1,seed=NULL,window=10,increment=1)
  trdata<-datos[obtsamples$tr,] #tomo datos para training
  testdata<-datos[obtsamples$ts,] #tomo datos para test
  
  RN.fit<-nnet(filtvbles, data=trdata,size=s,maxit=m,decay=d)#estimo modelo
  
  vpredrn1<-predict(RN.fit, trdata,type="raw")#prediccion sobre training data
  for(j in 1:5){
    repres1<-Computeauc(trdata,vpredrn1)
    listauc1<-c(listauc1,repres1)
    
  }
  
  res1obj.rn<-mean(unlist(listauc1))
  
  
  
  vpredrn2<-predict(RN.fit, testdata,type="raw")#prediccion sobre test data
  for(z in 1:5){
  repres2<-Computeauc(testdata,vpredrn2)
  listauc2<-c(listauc2,repres2)
 
  }
  
  res2obj.rn<-mean(unlist(listauc2))
  
  result<-c(res1obj.rn,res2obj.rn) #ambos resultados a un vector 1-training 2-test
  
  return(result)
}


RHoldOutRN_5AUC<-function(datos,n,esq.val,s,m,d){
  replicate(n,HoldOutRN_5AUC(datos,esq.val,s,m,d))
}


Computeauc<-function(data,vpredrn2){
  res<-auc(data$Class,vpredrn2)
  return(res)
}

res_5AUC<-RHoldOutRN_5AUC(data,30,2/3,5,100,4e-1)
res_150<-RHoldOutRN(data,150,2/3,5,100,4e-1)

mean(res_5AUC[1,])
mean(res_5AUC[2,]) 
mean(res_150[1,])
mean(res_150[2,]) 

par(mfrow=c(1,2))
boxplot(cbind(unlist(res_5AUC[1,]),unlist(res_5AUC[2,])),xlab="AUC 5 veces y RHoldout 30 veces",ylim=c(0.4,1.5))
boxplot(cbind(unlist(res_150[1,]),unlist(res_150[2,])),xlab="RHoldout 150 veces",ylim=c(0.4,1.5))

# EN REALIDAD SOLO NOS INTERESA AUC HOLDOUT EN GENERALIZACION.


#vemos que se obtienen los mismos resultados 

#------------------------
#metodo wrapper---------


CrValLRW<-function (subset){
  folds <- createFolds(data$Class, k=10)
  resultt<- list();
  restest <- list();
  
  for(i in 1:10){
    testData <- data[unlist(folds[i]), ] #divido la primera vez
    trainData <- data[-unlist(folds[i]), ]
    
    rl.fit<-glm(as.simple.formula(subset,"Class"), data=trainData,family=binomial("logit")) # modelo con subset
    res<-ComputeAUCLogRegr (rl.fit,testData) #calculo auc del modelo
    restest<-c(restest,res) #meto auc en lista 
  }
  
  resultt<-c(resultt,restest)
  
  return(mean(unlist(resultt)))
}




ComputeAUCLogRegr <- function(fittedmodel,testData) {
  vpred1<-predict(fittedmodel, testData,type="response")
  
  obj.rl1<-auc(testData$Class,vpred1)
  
  return (obj.rl1)
}




start.time.wrapper <- Sys.time()
subset<-forward.search(names(data)[-4027],CrValLRW)
end.time.wrapper<- Sys.time()
time.taken.wrapper<-end.time.wrapper - start.time.wrapper

fw<-as.simple.formula(subset,"Class")
print(fw)






resWrapper<-ValCrDT_WRAPPER(data,0.06)

resFilter<-ValCrDT(data,0.06)



par(mfrow=c(1,2))
boxplot(cbind(unlist(resFilter[1,]),unlist(resFilter[2,])),xlab="resFilter",ylim=c(0.4,1.5))
boxplot(cbind(unlist(resWrapper[1,]),unlist(resWrapper[2,])),xlab="resWrapper",ylim=c(0.4,1.5))


ValCrDT_WRAPPER<-function (datos,cp){
  folds <- createFolds(datos$Class, k=10)
  restraining <- list();
  restest <- list();
  for(i in 1:10){
    testData <- datos[unlist(folds[i]), ]
    trainData <- datos[-unlist(folds[i]), ]
    
    
    
    Dt.fit<-rpart(fw, data=trainData,control=rpart.control(cp=cp))
    
    Dt.pred1<-predict(Dt.fit,trainData,type="prob") 
    
    res1obj.dt<-auc(trainData$Class,Dt.pred1[,"germinal"])
    
    
    Dt.pred2<-predict(Dt.fit,testData,type="prob") 
    
    res2obj.dt<-auc(testData$Class,Dt.pred2[,"germinal"])
    
    restraining<-c(restraining,res1obj.dt)
    restest<-c(restest,res2obj.dt)
  }
  result<-matrix(ncol=length(restest),nrow=2)
  
  result[1,]<-unlist(restraining); #ambos resultados a un vector 1-training 2-test
  result[2,]<-unlist(restest);
  return(result)
}


#Vble categorica y vble numerica, hacemos T.test (distribucion normal)
#-------------GENE3261X + GENE3327X + GENE3330X + GENE3328X
#p<0.4? se rechaza hip nula en t test. Se considera entonces que NO son independientes.
t.test(GENE3261X~Class,data)
# p-value = 3.994e-09
t.test(GENE3327X~Class,data)
t.test(GENE3330X~Class,data)
t.test(GENE3328X~Class,data)

#GEN AL AZAR:

t.test(GENE1809X~Class,data)

#p-value = 0.8622  no se rechaza la hip nula, se puede decir q son independientes.
