setwd("/projects/p31385/pain-prediction")

library(psych)              # biscuit / biscwit
library(glmnet)             # elastic net regression
library(glmnetUtils)        # extension of basic elastic net with CV
library(caret)              # train and test for random forest
library(vip)                # variable importance
library(lubridate)          # date wrangling
library(plyr)               # data wranging
library(tidyverse)          # data wrangling
library(cowplot)            # flexibly arrange multiple ggplot objects
library(tidymodels)         # tidy model workflow and selection
# library(modeltime)          # tidy models for time series
# library(furrr)              # mapping many models in parallel 

sessionInfo()

jobid = as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))+1
print(jobid)
args <- read.csv("scripts/cluster/args/glmnet.csv"
                   , header = T, stringsAsFactors = F)[jobid,]
print(args)

dummy_vars <- c("caregiving", "chores", "exercise", "internet", "mentalAct", "nothing", "otherAct"
                , "selfcare", "socialOnline", "TV", "volunteer", "reclining", "sitting", "standing"
                , "acquaintance", "alone", "family", "friend", "kids", "neighbor","otherPerson"
                , "partner", "pet", "stranger", "socialPerson", "media", "activity", "social")
time_dummy <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
                , "morning", "midday", "evening", "night")
time_vars <- c("cos1p", "cos2p", "cub", "linear", "quad", "sin1p", "sin2p")

c_fun <- function(m){
  # final model characteristics
  lambda <- min(m$fit$lambda)
  coefs <- stats::coef(m$fit, s = lambda)
  coefs <- coefs[, 1L, drop = TRUE]
  coefs <- coefs[setdiff(x = names(coefs), y = "(Intercept)")]
  return(coefs)
}

elnet_fun <- function(sid, outcome, group, time){
  # load the data
  load(sprintf("02-data/03-train-data/%s_%s_%s_%s.RData",
               sid, outcome, group, time))
  d_train <- d_train %>% arrange(Full_Date) %>% select(-Full_Date)
  
  d_train_cv <- rolling_origin(
    d_train, 
    initial = 15, 
    skip = 1,
    assess = 3,
    cumulative = F
  )
  
  # set up the cross-valiation folds
  # set.seed(234)
  # d_train_cv <- vfold_cv(d_train, v = 10)
  
  # set up the data and formula
  mod_recipe <- recipe(
    o_value ~ .
    , data = d_train
  ) %>%
    step_zv(all_numeric()) %>%
    step_normalize(all_numeric(), -one_of(time_vars)) %>%
    step_nzv(one_of(dummy_vars, time_dummy), -all_outcomes()) %>%
    step_dummy(one_of(dummy_vars, time_dummy), -all_outcomes()) #%>%
    # estimate the means and standard deviations
    # prep(training = d_train, retain = TRUE)
  
  # set up the model specifications 
  tune_spec <- 
    linear_reg(
      penalty = tune()
      , mixture = tune()
    ) %>% 
    set_engine("glmnet") %>% 
    set_mode("regression")
  
  # set up the ranges for the tuning functions 
  elnet_grid <- 
    grid_regular(
      penalty()
      , mixture()
      , levels = 10
    )
  
  # set up the workflow: combine modeling spec with modeling recipe
  set.seed(345)
  elnet_wf <- workflow() %>%
    add_model(tune_spec) %>%
    add_recipe(mod_recipe)
  
  # combine the workflow, and grid to a final tuning model
  elnet_res <- 
    elnet_wf %>% 
    tune_grid(
      resamples = d_train_cv
      , grid = elnet_grid
      , control = control_resamples(save_pred = T)
    )
  save(elnet_res, file = sprintf("05-results/01-glmnet/01-tuning-models/%s_%s_%s_%s.RData",
                                 sid, outcome, group, time))
  
  # plot the metrics across tuning parameters
  # p <- elnet_res %>%
  #   collect_metrics() %>%
  #   ggplot(aes(penalty, mean, color = mixture)) +
  #   geom_point(size = 2) +
  #   facet_wrap(~ .metric, scales = "free", nrow = 2) +
  #   scale_x_log10(labels = scales::label_number()) +
  #   scale_color_gradient(low = "gray90", high = "red") +
  #   theme_classic()
  # ggsave(p, file = sprintf("05-results/01-glmnet/02-tuning-figures/%s_%s_%s_%s.png",
  #                          sid, outcome, group, time)
  #        , width = 5, height = 8)
  
  # select the best model based on AUC
  best_elnet <- elnet_res %>%
    # select_best("roc_auc")
    select_best("rmse")
  
  # set up the workflow for the best model
  final_wf <- 
    elnet_wf %>% 
    finalize_workflow(best_elnet)
  
  # run the final best model on the training data and save
  final_elnet <- 
    final_wf %>%
    fit(data = d_train) 
  
  final_m <- final_elnet %>% 
    pull_workflow_fit() 
  
  final_coefs <- c_fun(final_m)
  
  best_elnet <- best_elnet %>%
    mutate(nvars = length(final_coefs[final_coefs != 0]))
  
  save(final_coefs, best_elnet,
       file = sprintf("05-results/01-glmnet/07-final-model-param/%s__%s_%s_%s.RData",
                      sid, outcome, group, time))
  
  # load the split data
  load(sprintf("02-data/04-test-data/%s_%s_%s_%s.RData",
               sid, outcome, group, time))
  # d_split$data$o_value <- factor(d_split$data$o_value)
  
  # run the final fit workflow of the training and test data together
  final_fit <- 
    final_wf %>%
    last_fit(d_split) 
  save(final_elnet, final_fit
       , file = sprintf("05-results/01-glmnet/03-final-training-models/%s_%s_%s_%s.RData",
                        sid, outcome, group, time))
  
  # final metrics (accuracy and roc)
  final_metrics <- final_fit %>%
    collect_metrics(summarize = T)
  save(final_metrics
       , file = sprintf("05-results/01-glmnet/06-final-model-performance/%s_%s_%s_%s.RData",
                        sid, outcome, group, time))
  
  # variable importance
  final_var_imp <- final_elnet %>% 
    extract_fit_parsnip() %>% 
    vi() %>%
    slice_max(Importance, n = 10)
  save(final_var_imp
       , file = sprintf("05-results/01-glmnet/05-variable-importance/%s_%s_%s_%s.RData",
                        sid, outcome, group, time))
  
  # # roc plot
  # p_roc <- final_fit %>%
  #   collect_predictions() %>% 
  #   roc_curve(.pred, truth = o_value) %>% 
  #   autoplot() + 
  #   labs(title = sprintf("Participant %s: %s, %s, %s, %s"
  #                        , sid, outcome, group, set, time)) 
  # ggsave(p_roc, file = sprintf("05-results/01-glmnet/04-roc-curves/%s_%s_%s_%s_%s.png",
  #              sid, outcome, group, set, time)
  #        , width = 5, height = 5)
  
  rm(list = c("final_var_imp", "final_metrics", "final_wf", "final_elnet", "final_fit"
              , "best_elnet", "elnet_res", "elnet_wf", "elnet_grid", "tune_spec", "mod_recipe"
              , "p", "p_roc", "d_split", "d_test", "d_train", "d_train_cv"))
  gc()
  return(T)
}

elnet_fun(args[,1], args[,2], args[,3], args[,4])