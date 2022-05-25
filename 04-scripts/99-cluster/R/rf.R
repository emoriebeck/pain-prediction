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
args <- read.csv("scripts/cluster/args/rf.csv"
                   , header = T, stringsAsFactors = F)[jobid,]
print(args)

dummy_vars <- c("caregiving", "chores", "exercise", "internet", "mentalAct", "nothing", "otherAct"
                , "selfcare", "socialOnline", "TV", "volunteer", "reclining", "sitting", "standing"
                , "acquaintance", "alone", "family", "friend", "kids", "neighbor","otherPerson"
                , "partner", "pet", "stranger", "socialPerson", "media", "activity", "social")

time_dummy <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
                , "morning", "midday", "evening", "night") 

time_vars <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
               , "morning", "midday", "evening", "night"
               , "sin2p", "sin1p", "cos2p", "cos1p"
               , "cub", "linear", "quad")


rf_fun <- function(sid, outcome, group, time){
  load(sprintf("02-data/02-model-data/%s_%s_%s_%s.RData",
               sid, outcome, group, time))
  # differencing and box-cox
  d <- d %>%
    mutate_if(is.factor, ~as.numeric(as.character(.))) %>%
    mutate_at(vars(-one_of(c(dummy_vars, time_vars)), -o_value, -Full_Date), ~ . + 1) %>%
    mutate_at(vars(-one_of(c(dummy_vars, time_vars)), -o_value, -Full_Date), log) %>% 
    mutate_at(vars(-one_of(c(dummy_vars, time_vars)), -o_value, -Full_Date), ~. - lag(.))
  # time delay embedding
  d_mbd <- map(d %>% select(-Full_Date, -one_of(time_vars)), ~embed(., 2)) %>% ldply(.) %>%
    group_by(.id) %>%
    mutate(beep = 1:n()) %>%
    ungroup() %>%
    select(-`1`) %>%
    pivot_wider(names_from = ".id"
                # , names_glue = "{.id}_{.value}"
                # , values_from = c("1", "2")
                , values_from = "2"
    ) %>%
    bind_cols(d[-1,] %>% select(Full_Date)) %>%
    select(-beep)
  
  d_mbd <- d_mbd %>% 
    full_join(d %>% select(Full_Date, one_of(time_vars))) %>%
    mutate_at(vars(contains(dummy_vars)), factor) %>%
    # mutate(o_value_1 = factor(o_value_1)) %>%
    drop_na()
  
  # training and test sets
  d_split <- initial_time_split(d_mbd, prop = 0.75)
  d_train <- training(d_split)
  d_test  <- testing(d_split)
  
  d_train <- d_train %>% arrange(Full_Date) %>% select(-Full_Date)
  
  ## create the rolling_origin training and validation sets
  init <- ceiling(nrow(d_train)/3)
  
  # set up the cross-valiation folds
  d_train_cv <- rolling_origin(
    d_train, 
    initial = 15, 
    skip = 1,
    assess = 3,
    cumulative = F
  )
  
  # set up the data and formula
  mod_recipe <- recipe(
    o_value ~ .
    , data = d_train
  ) %>%
    step_zv(all_numeric(), contains(dummy_vars), contains(time_vars)) %>%
    step_dummy(one_of(c(time_dummy, dummy_vars)), -all_outcomes()) %>%
    step_nzv(all_predictors(), unique_cut = 35) #%>%
  # estimate the means and standard deviations
  # prep(training = d_train, retain = TRUE)
  
  # set up the model specifications 
  tune_spec <- 
    rand_forest(
      mtry = tune()
      , trees = 1000
      , min_n = tune()
    ) %>% 
    set_engine("ranger", importance = "permutation") %>%
    set_mode("regression")
  
  # set up the workflow: combine modeling spec with modeling recipe
  set.seed(345)
  rf_wf <- workflow() %>%
    add_model(tune_spec) %>%
    add_recipe(mod_recipe)
  
  # set up the ranges for the tuning functions 
  set.seed(345)
  tune_res <- tune_grid(
    rf_wf
    , resamples = d_train_cv
    , grid = 20
  )
  save(tune_res, file = sprintf("05-results/03-rf/01-tuning-models/%s_%s_%s_%s.RData",
                                sid, outcome, group, time))
  
  # load(sprintf("05-results/03-rf/01-tuning-models/%s_%s_%s_%s_%s.RData",
  #              sid, outcome, group, set, time))
  
  # plot the metrics across tuning parameters
  p <- tune_res %>%
    collect_metrics() %>%
    ggplot(aes(mtry, mean, color = min_n)) +
    geom_point(size = 2) +
    facet_wrap(~ .metric, scales = "free", nrow = 2) +
    scale_x_log10(labels = scales::label_number()) +
    scale_color_gradient(low = "gray90", high = "red") +
    theme_classic()
  ggsave(p, file = sprintf("05-results/03-rf/02-tuning-figures/%s_%s_%s_%s.png",
                           sid, outcome, group, time)
         , width = 5, height = 8)
  
  # select the best model based on AUC
  best_rf <- tune_res %>%
    # select_best("roc_auc")
    select_best("rmse")
  
  # set up the workflow for the best model
  final_wf <- 
    rf_wf %>% 
    finalize_workflow(best_rf)
  
  # run the final best model on the training data and save
  final_rf <- 
    final_wf %>%
    fit(data = d_train) 
  
  final_m <- final_rf %>% 
    extract_fit_parsnip()
  final_coefs <- final_m$fit$variable.importance
  
  best_rf <- best_rf %>%
    mutate(nvars = length(final_coefs[final_coefs != 0]))
  
  save(final_coefs, best_rf,
       file = sprintf("05-results/03-rf/07-final-model-param/%s_%s_%s_%s.RData",
                      sid, outcome, group, time))
  
  # run the final fit workflow of the training and test data together
  final_fit <- 
    final_wf %>%
    last_fit(d_split) 
  save(final_rf, final_fit
       , file = sprintf("05-results/03-rf/03-final-training-models/%s_%s_%s_%s.RData",
                        sid, outcome, group, time))
  
  # final metrics (accuracy and roc)
  final_metrics <- final_fit %>%
    collect_metrics(summarize = T)
  save(final_metrics
       , file = sprintf("05-results/03-rf/06-final-model-performance/%s_%s_%s_%s.RData",
                        sid, outcome, group, time))
  
  # variable importance
  final_var_imp <- final_rf %>% 
    extract_fit_parsnip() %>% 
    vi() %>%
    slice_max(Importance, n = 10)
  save(final_var_imp
       , file = sprintf("05-results/03-rf/05-variable-importance/%s_%s_%s_%s.RData",
                        sid, outcome, group, time))
  
  # roc plot
  # p_roc <- final_fit %>%
  #   collect_predictions() %>% 
  #   roc_curve(.pred_0, truth = o_value_1) %>% 
  #   autoplot() + 
  #   labs(title = sprintf("Participant %s: %s, %s, %s, %s"
  #                        , sid, outcome, group, set, time)) 
  # ggsave(p_roc, file = sprintf("05-results/03-rf/04-roc-curves/%s_%s_%s_%s_%s.png",
  #              sid, outcome, group, set, time)
  #        , width = 5, height = 5)
  
  rm(list = c("final_var_imp", "final_metrics", "final_wf", "final_rf", "final_fit"
              , "best_rf", "tune_res", "rf_wf", "tune_spec", "mod_recipe"
              , "p", "p_roc", "d_split", "d_test", "d_train", "d_train_cv"))
  gc()
  return(T)
}

rf_fun(args[,1], args[,2], args[,3], args[,4])