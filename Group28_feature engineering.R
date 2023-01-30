#load libraries
library(tidyverse)
library(caret)
library(stringi)
library(timeDate)
library(text2vec)
library(tm)
library(SnowballC)
library(glmnet)
library(vip)
library(naivebayes)
library(ranger)
library(xgboost)

#======================load data========================

#load data files and label the data (train & test)
train_x <- read_csv("ks_training_X.csv") %>%
  mutate(is_train = 1)

test_x <- read_csv("ks_test_X.csv") %>%
  mutate(is_train = 0)

train_y <- read_csv("ks_training_y.csv")

# Stack training/labeled and testing/unlabeled so features can be processing at the same time 
projects <- rbind(train_x, test_x)


#=======================features engineering========================

# Use external data - Social security data on gender of babies born by their first names
names = read_csv("baby_names.csv") %>%
  mutate(creator_firstname = toupper(name),
         creator_female_name = ifelse(percent_female >= .5, 1, 0)) %>%
  select(creator_firstname, creator_female_name)

projects <- projects %>%
  mutate(launch_period = as.double(deadline - launched_at, units = "days")) %>% # how long the project has been launched
  mutate(launch_on_weekend = isWeekend(launched_at),
         launch_on_weekend = as.factor(case_when(
           launch_on_weekend == TRUE ~ "YES",
           launch_on_weekend == FALSE ~ "NO"
         ))) %>% # check whether a project is launched on weekend
  separate(created_at, sep = "-", into = c("create_year", "create_month", "create_date")) %>% # separate created date into year, month, and date # remove = FALSE ??
  separate(launched_at, sep = "-", into = c("launch_year", "launch_month", "launch_date")) %>% # separate launch date into year, month, and date # remove = FALSE ??
  separate(deadline, sep = "-", into = c("deadline_year", "deadline_month", "deadline_date")) %>% # separate deadline into year, month, and date # remove = FALSE ??
  mutate(create_year = as.numeric(create_year),
         launch_year = as.numeric(launch_year),
         deadline_year = as.numeric(deadline_year)) %>% # convert years into number
  mutate(create_month = as.factor(create_month),
         launch_month = as.factor(launch_month),
         deadline_month = as.factor(deadline_month)) %>% # convert months into factor
  mutate(create_date = as.numeric(create_date),
         launch_date = as.numeric(launch_date),
         deadline_date = as.numeric(deadline_date)) %>% # convert dates into numbers
  mutate(create_date = as.factor(case_when(
    create_date >= 1 & create_date <= 10 ~ "Early",
    create_date >= 11 & create_date <= 20 ~ "Mid",
    create_date >= 21 & create_date <= 31 ~ "Late"
  ))) %>% # convert numeric dates into three groups: early, mid, and late
  mutate(launch_date = as.factor(case_when(
    launch_date >= 1 & launch_date <= 10 ~ "Early",
    launch_date >= 11 & launch_date <= 20 ~ "Mid",
    launch_date >= 21 & launch_date <= 31 ~ "Late"
  ))) %>% # convert numeric dates into three groups: early, mid, and late
  mutate(deadline_date = as.factor(case_when(
    deadline_date >= 1 & deadline_date <= 10 ~ "Early",
    deadline_date >= 11 & deadline_date <= 20 ~ "Mid",
    deadline_date >= 21 & deadline_date <= 31 ~ "Late"
  ))) %>% # convert numeric dates into three groups: early, mid, and late
  mutate(location_state = str_sub(location_slug, -2, -1),
         location_state = as.factor(toupper(location_state))) %>% # extract the location state from the location slug and capitalized it
  mutate(location_place = str_sub(location_slug, 1, -4),
         location_place = as.factor(location_place)) %>% # extract the location place from the location slug
  separate(creator_name, into = ("creator_firstname"), extra = "drop") %>%
  mutate(creator_firstname = toupper(creator_firstname)) %>%
  left_join(names, by = "creator_firstname") %>%
  mutate(creator_female_name = case_when(
    creator_female_name == 1 ~ "F",
    creator_female_name == 0 ~ "M",
    is.na(creator_female_name) ~  "U"),
    creator_female_name = as.factor(creator_female_name)) %>% # assign gender to creators
  mutate(goal_quantile_rank = ntile(goal, 4),
         big_project = as.factor(ifelse(goal_quantile_rank == 4, "YES", "NO"))) %>% # rank project goal and create "big_project"
  mutate(name_word_count = str_count(name, "\\w+"),
         name_word_count = ifelse(is.na(name_word_count), 0, name_word_count)) %>% # count the number of words in the project name
  mutate(isbwImg1 = as.factor(case_when(
    isbwImg1 == FALSE ~ "NO",
    isbwImg1 == TRUE ~ "YES",
    TRUE ~ "UNKNOWN"
  ))) %>% # convert isbwImg1 from a logical variable into factors and create a thrid group for NAs
  mutate(isTextPic = as.factor(case_when(
    isTextPic == 0 ~ "NO",
    isTextPic == 1 ~ "YES",
    TRUE ~ "UNKNOWN"
  )),
  isLogoPic = as.factor(case_when(
    isLogoPic == 0 ~ "NO",
    isLogoPic == 1 ~ "YES",
    TRUE ~ "UNKNOWN"
  )),
  isCalendarPic = as.factor(case_when(
    isCalendarPic == 0 ~ "NO",
    isCalendarPic == 1 ~ "YES",
    TRUE ~ "UNKNOWN"
  )),
  isDiagramPic = as.factor(case_when(
    isDiagramPic == 0 ~ "NO",
    isDiagramPic == 1 ~ "YES",
    TRUE ~ "UNKNOWN"
  )),
  isShapePic = as.factor(case_when(
    isShapePic == 0 ~ "NO",
    isShapePic == 1 ~ "YES",
    TRUE ~ "UNKNOWN"
  ))) %>% # convert NAs into UNKNOWN
  mutate(smiling_project = ifelse(smiling_project > 100, 100, smiling_project),
         smiling_project = round(smiling_project, digits = 0)) %>% # for probability > 100, transfrom to max = 100 and round it up
  mutate(smiling_creator = ifelse(smiling_creator > 100, 100, smiling_creator),
         smiling_creator = round(smiling_creator, digits = 0)) %>% # for probability > 100, transfrom to max = 100 and round it up
  mutate(reward_amounts = ifelse(is.na(reward_amounts), 0, reward_amounts),
         reward_amount_count = str_count(reward_amounts, ","),
         reward_amount_count = ifelse(is.na(reward_amount_count), 0, reward_amount_count)) %>% # count the number of reward donation options
  mutate(min_reward_amount = as.numeric(stri_match_first_regex(reward_amounts, "\\d+\\.\\d")),
         min_reward_amount = ifelse(is.na(min_reward_amount), 0, min_reward_amount)) %>% # identify the min value of reward_amounts
  mutate(max_reward_amount = as.numeric(stri_match_last_regex(reward_amounts, "\\d+\\.\\d")),
         max_reward_amount = ifelse(is.na(max_reward_amount), 0, max_reward_amount)) %>% # identify the max value of reward_amounts
  mutate(tag_count = str_count(tag_names, "\\w+"),
         tag_count = ifelse(is.na(tag_count), 0, tag_count)) %>% # count the number of tag names
  group_by(category_parent) %>%
  mutate(category_freq = n()) %>%
  ungroup() %>%
  mutate(is_popular_cat = ifelse(category_freq < nrow(projects) / length(unique(category_parent)), "NO", "YES"),
         is_popular_cat = as.factor(ifelse(is.na(is_popular_cat), "UNKNOWN", is_popular_cat))) %>% # determine whether a project category is popular (if each cat accounts for the avg)
  mutate(minage_project_group = case_when(
    minage_project >= 1 & minage_project <= 4 ~ "Toddler",
    minage_project >= 5 & minage_project <= 12 ~ "Gradeschooler",
    minage_project >= 13 & minage_project <= 17 ~ "Teen",
    minage_project >= 18 & minage_project <= 35 ~ "Young adult",
    minage_project >= 36 & minage_project <= 55 ~ "Mid-aged",
    minage_project >= 56 ~ "Elder",
    TRUE ~ "None"),
    minage_project_group = as.factor(minage_project_group)) %>% # use age group instead of ages for min age of projects
  mutate(minage_creator_group = case_when(
    minage_creator >= 1 & minage_creator <= 4 ~ "Toddler",
    minage_creator >= 5 & minage_creator <= 12 ~ "Gradeschooler",
    minage_creator >= 13 & minage_creator <= 17 ~ "Teen",
    minage_creator >= 18 & minage_creator <= 35 ~ "Young adult",
    minage_creator >= 36 & minage_creator <= 55 ~ "Mid-aged",
    minage_creator >= 56 ~ "Elder",
    TRUE ~ "None"),
    minage_creator_group = as.factor(minage_creator_group)) %>% # use age group instead of ages for min age of creators
  mutate(maxage_project_group = case_when(
    maxage_project >= 1 & maxage_project <= 4 ~ "Toddler",
    maxage_project >= 5 & maxage_project <= 12 ~ "Gradeschooler",
    maxage_project >= 13 & maxage_project <= 17 ~ "Teen",
    maxage_project >= 18 & maxage_project <= 35 ~ "Young adult",
    maxage_project >= 36 & maxage_project <= 55 ~ "Mid-aged",
    maxage_project >= 56 ~ "Elder",
    TRUE ~ "None"),
    maxage_project_group = as.factor(maxage_project_group)) %>% # use age group instead of ages for max age of projects
  mutate(maxage_creator_group = case_when(
    maxage_creator >= 1 & maxage_creator <= 4 ~ "Toddler",
    maxage_creator >= 5 & maxage_creator <= 12 ~ "Gradeschooler",
    maxage_creator >= 13 & maxage_creator <= 17 ~ "Teen",
    maxage_creator >= 18 & maxage_creator <= 35 ~ "Young adult",
    maxage_creator >= 36 & maxage_creator <= 55 ~ "Mid-aged",
    maxage_creator >= 56 ~ "Elder",
    TRUE ~ "None"),
    maxage_creator_group = as.factor(maxage_creator_group)) %>% # use age group instead of ages for max age of creators
  mutate(color_foreground = as.factor(ifelse(is.na(color_foreground), "Unknown", color_foreground)),
         color_background = as.factor(ifelse(is.na(color_background), "Unknown", color_background))) %>% # clean up features: color_fore/background
  group_by(location_type) %>%
  mutate(location_type_freq = n()) %>%
  ungroup() %>%
  mutate(location_type =  as.factor(ifelse(location_type_freq < 100, "Other", location_type))) %>% # reassign location_type 
  mutate(contains_youtube = ifelse(contains_youtube == 0, "NO", "YES"),
         contains_youtube = as.factor(contains_youtube),
         region = as.factor(region),
         category_parent = as.factor(category_parent)) %>% # convert character vars into factors
  mutate(descriptionPositive = case_when(
    afinn_pos - afinn_neg > 0 ~ "YES",
    afinn_pos - afinn_neg < 0 ~ "NO",
    afinn_pos - afinn_neg == 0 ~ "NEUTRAL"
  ),
    descriptionPositive = as.factor(ifelse(is.na(descriptionPositive), "UNKNOWN", descriptionPositive))) %>% # identify whether the description has positive/negative sentiment
  mutate(tag_names = str_replace_all(tag_names, " ", ""), # remove white space so that tags with multiple words can be considered as one single tag
         tag_names = str_replace_all(tag_names, "\\|", " ")) %>% # remove "|" so we treat each tag as one vocabulary
  mutate(ADV_percentage = round(ADV / num_words, 4),
         NOUN_percentage = round(NOUN / num_words, 4),
         ADP_percentage = round(ADP / num_words, 4),
         PRT_percentage = round(PRT / num_words, 4),
         DET_percentage = round(DET / num_words, 4),
         PRON_percentage = round(PRON / num_words, 4),
         VERB_percentage = round(VERB / num_words, 4),
         NUM_percentage = round(NUM / num_words, 4),
         CONJ_percentage = round(CONJ / num_words, 4),
         ADJ_percentage = round(ADJ / num_words, 4)) %>% # Calculate the percentage of each type of words in the full description
  mutate(ADV_percentage = ifelse(is.na(ADV_percentage), 0, ADV_percentage),
         NOUN_percentage = ifelse(is.na(NOUN_percentage), 0, NOUN_percentage),
         ADP_percentage = ifelse(is.na(ADP_percentage), 0, ADP_percentage),
         PRT_percentage = ifelse(is.na(PRT_percentage), 0, PRT_percentage),
         DET_percentage = ifelse(is.na(DET_percentage), 0, DET_percentage),
         PRON_percentage = ifelse(is.na(PRON_percentage), 0, PRON_percentage),
         VERB_percentage = ifelse(is.na(VERB_percentage), 0, VERB_percentage),
         NUM_percentage = ifelse(is.na(NUM_percentage), 0, NUM_percentage),
         CONJ_percentage = ifelse(is.na(CONJ_percentage), 0, CONJ_percentage),
         ADJ_percentage = ifelse(is.na(ADJ_percentage), 0, ADJ_percentage)) # remove NAs and replace it with 0





#============================TEXT MINING ON BLURB================================

# Retrieve the project id and their corresponding blurb from the labeled data --> create vocabularies
blurb_labeled <- projects %>%
  filter(is_train == 1) %>%
  select(id, blurb) %>%
  left_join(train_y, by = "id") %>%
  mutate(success_numeric = ifelse(success == "NO", 0, 1),
         success = as.factor(ifelse(success == "NO", 0, 1))) %>%
  select(id, blurb, success, success_numeric)

# Define a tokenizer
prep = tolower 

blurb_tokenizer <- function(v){
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    removeWords(stopwords(kind="en")) %>% #remove stopwords
    stemDocument %>%
    word_tokenizer 
}

blurb_tok = blurb_tokenizer

# Convert individual documents into tokens with the function defined above
it_blurb_labeled = itoken(blurb_labeled$blurb,
                        preprocessor = prep,
                        tokenizer = blurb_tok,
                        ids = blurb_labeled$id,
                        progressbar = FALSE)

# Create a list of stop words and create vocabs from blurb
stop_words = c("will", "make", "need", "one", "us", "get", "can", "want", "show", "bring", "take", "use", "two", "come", "look", "go", "made", "help")
blurb_vocab = create_vocabulary(it_blurb_labeled, stopwords = stop_words, ngram = c(1L, 2L))
blurb_vocab = prune_vocabulary(blurb_vocab, doc_count_min = 500)

# Vectorize the vocab
blurb_vectorizer = vocab_vectorizer(blurb_vocab)

dtm_blurb_labeled = create_dtm(it_blurb_labeled, blurb_vectorizer)

tfidf = TfIdf$new()
tfidf_blurb_labeled = fit_transform(dtm_blurb_labeled, tfidf)

#split labeled into train/valid
set.seed(1)
train_rows_blurb <- sample(nrow(blurb_labeled),.7*nrow(blurb_labeled))
tr_tfidf_blurb <- tfidf_blurb_labeled[train_rows_blurb,]
va_tfidf_blurb <- tfidf_blurb_labeled[-train_rows_blurb,]

# Get the y values
tr_y_blurb <- blurb_labeled[train_rows_blurb,]$success
va_y_blurb <- blurb_labeled[-train_rows_blurb,]$success
tr_y_numeric_blurb <- blurb_labeled[train_rows_blurb,]$success_numeric
va_y_numeric_blurb <- blurb_labeled[-train_rows_blurb,]$success_numeric

# Train Ridge and Lasso models to generate accuracy
grid <- 10^seq(7,-7,length=100)
k<-5

cv.out.ridge.blurb <- cv.glmnet(tr_tfidf_blurb, tr_y_blurb, family="binomial", alpha = 0, lambda = grid, nfolds = k)
bestlam_ridge_blurb <- cv.out.ridge.blurb$lambda.min
pred_ridge_blurb <- predict(cv.out.ridge.blurb, s=bestlam_ridge_blurb, newx = va_tfidf_blurb, type="response")
class_ridge_blurb <- ifelse(pred_ridge_blurb > 0.5, "1", "0")
acc_ridge_blurb <- mean(ifelse(class_ridge_blurb == va_y_blurb, 1, 0))
acc_ridge_blurb

cv.out.lasso.blurb <- cv.glmnet(tr_tfidf_blurb, tr_y_blurb, family="binomial", alpha = 1, lambda = grid, nfolds = k)
bestlam_lasso_blurb <- cv.out.lasso.blurb$lambda.min
pred_lasso_blurb <- predict(cv.out.lasso.blurb, s=bestlam_lasso_blurb, newx = va_tfidf_blurb, type="response")
class_lasso_blurb <- ifelse(pred_lasso_blurb > 0.5, "1", "0")
acc_lasso_blurb <- mean(ifelse(class_lasso_blurb == va_y_blurb, 1, 0))
acc_lasso_blurb

# Train a random forest model to generate an accuracy
rf.mod.blurb <- ranger(x = tr_tfidf_blurb, y = tr_y_numeric_blurb,
                 mtry = 15, num.trees = 500,
                 importance="impurity",
                 probability = TRUE)

pred_rf_blurb <- predict(rf.mod.blurb, data=va_tfidf_blurb)$predictions[,2]
class_rf_blurb <- ifelse(pred_rf_blurb > 0.5, 1, 0)
acc_rf_blurb <- mean(ifelse(class_rf_blurb == va_y_numeric_blurb, 1, 0))
acc_rf_blurb

# Train a boosting model to generate an accuracy
bst.mod.blurb <- xgboost(data = tr_tfidf_blurb, label = tr_y_numeric_blurb, max.depth = 2, eta = 1, nrounds = 1000,  objective = "binary:logistic")

pred_bst_blurb <- predict(bst.mod.blurb, va_tfidf_blurb)
class_bst_blurb <- ifelse(pred_bst_blurb > 0.5, 1, 0)
acc_bst_blurb <- mean(ifelse(class_bst_blurb == va_y_numeric_blurb, 1, 0))
acc_bst_blurb

# Check the top-15 vocabs from the all models and select the most valuable ones manually
vip(cv.out.ridge.blurb, num_features = 15)
vip(cv.out.lasso.blurb, num_features = 15)
vip(rf.mod.blurb, num_features = 15)
vip(bst.mod.blurb, num_features = 15)

blurb_vocab_list <- c("theater", "theatr", "album", "danc", "art", "festiv", "music", "film", "game", "book", "design", "premier", "stori", "app", "cd")

# Retrieve blurbs from all data
blurb_all <- projects %>%
  select(id, blurb)

# Convert the blurb of entire dataset (both labeled and unlabeled) into sparse matrix
it_blurb_all = itoken(blurb_all$blurb,
                      preprocessor = prep,
                      tokenizer = blurb_tok,
                      ids = blurb_all$id,
                      progressbar = FALSE)

# Vectorize all blurb
dtm_blurb_all = create_dtm(it_blurb_all, blurb_vectorizer)
dtm_blurb_all_bin <- dtm_blurb_all>0+0

# Select out those terms and convert them into a dataframe so it can be bind with the original dataset
# Modify the variable names
dtm_blurb_all_small <- dtm_blurb_all_bin[, blurb_vocab_list]
blurb_dense = as.matrix(dtm_blurb_all_small)+0
blurb_vocab_df = as.data.frame(blurb_dense)
blurb_vocab_df <- blurb_vocab_df %>%
  rename_with( ~ paste0("blurb_contains_", .x))

# Replace 0 with NO, 1 with YES
blurb_vocab_df[blurb_vocab_df == 0] <- "NO"
blurb_vocab_df[blurb_vocab_df == 1] <- "YES"



#============================TEXT MINING ON TAG_NAMES===============================

# Retrieve the project id and their corresponding blurb from the labeled data --> create vocabularies
tag_labeled <- projects %>%
  filter(is_train == 1) %>%
  select(id, tag_names) %>%
  left_join(train_y, by = "id") %>%
  mutate(success_numeric = ifelse(success == "NO", 0, 1),
         success = as.factor(ifelse(success == "NO", 0, 1))) %>%
  select(id, tag_names, success, success_numeric)

# Define a tokenizer
tag_tokenizer <- function(v){
  v %>%
    stemDocument %>%
    word_tokenizer 
}

tag_tok = tag_tokenizer

# Convert individual documents into tokens with the function defined above
it_tag_labeled = itoken(tag_labeled$tag_names,
                          preprocessor = prep,
                          tokenizer = tag_tok,
                          ids = tag_labeled$id,
                          progressbar = FALSE)

# Create vocabs from tag_names
tag_vocab = create_vocabulary(it_tag_labeled)
tag_vocab = prune_vocabulary(tag_vocab, doc_count_min = 100)

# Vectorize the vocab
tag_vectorizer = vocab_vectorizer(tag_vocab)

dtm_tag_labeled = create_dtm(it_tag_labeled, tag_vectorizer)

tfidf_tag_labeled = fit_transform(dtm_tag_labeled, tfidf)

#split labeled into train/valid
set.seed(1)
train_rows_tag <- sample(nrow(tag_labeled),.7*nrow(tag_labeled))
tr_tfidf_tag <- tfidf_tag_labeled[train_rows_tag,]
va_tfidf_tag <- tfidf_tag_labeled[-train_rows_tag,]

# Get the y values
tr_y_tag <- tag_labeled[train_rows_tag,]$success
va_y_tag <- tag_labeled[-train_rows_tag,]$success
tr_y_numeric_tag <- tag_labeled[train_rows_tag,]$success_numeric
va_y_numeric_tag <- tag_labeled[-train_rows_tag,]$success_numeric

# Train Ridge and Lasso models to generate accuracy
grid <- 10^seq(7,-7,length=100)
k<-5

cv.out.ridge.tag <- cv.glmnet(tr_tfidf_tag, tr_y_tag, family="binomial", alpha = 0, lambda = grid, nfolds = k)
bestlam_ridge_tag <- cv.out.ridge.tag$lambda.min
pred_ridge_tag <- predict(cv.out.ridge.tag, s=bestlam_ridge_tag, newx = va_tfidf_tag, type="response")
class_ridge_tag <- ifelse(pred_ridge_tag > 0.5, "1", "0")
acc_ridge_tag <- mean(ifelse(class_ridge_tag == va_y_tag, 1, 0))
acc_ridge_tag

cv.out.lasso.tag <- cv.glmnet(tr_tfidf_tag, tr_y_tag, family="binomial", alpha = 1, lambda = grid, nfolds = k)
bestlam_lasso_tag <- cv.out.lasso.tag$lambda.min
pred_lasso_tag <- predict(cv.out.lasso.tag, s=bestlam_lasso_tag, newx = va_tfidf_tag, type="response")
class_lasso_tag <- ifelse(pred_lasso_tag > 0.5, "1", "0")
acc_lasso_tag <- mean(ifelse(class_lasso_tag == va_y_tag, 1, 0))
acc_lasso_tag

# Train a random forest model to generate an accuracy
rf.mod.tag <- ranger(x = tr_tfidf_tag, y = tr_y_numeric_tag,
                       mtry = 15, num.trees = 500,
                       importance="impurity",
                       probability = TRUE)

pred_rf_tag <- predict(rf.mod.tag, data=va_tfidf_tag)$predictions[,2]
class_rf_tag <- ifelse(pred_rf_tag > 0.5, 1, 0)
acc_rf_tag <- mean(ifelse(class_rf_tag == va_y_numeric_tag, 1, 0))
acc_rf_tag

# Train a boosting model to generate an accuracy
bst.mod.tag <- xgboost(data = tr_tfidf_tag, label = tr_y_numeric_tag, max.depth = 2, eta = 1, nrounds = 1000,  objective = "binary:logistic")

pred_bst_tag <- predict(bst.mod.tag, va_tfidf_tag)
class_bst_tag <- ifelse(pred_bst_tag > 0.5, 1, 0)
acc_bst_tag <- mean(ifelse(class_bst_tag == va_y_numeric_tag, 1, 0))
acc_bst_tag

# Manually identify 15 the most useful tag names across multiple models 
vip(cv.out.ridge.tag, num_features = 15)
vip(cv.out.lasso.tag, num_features = 15)
vip(rf.mod.tag, num_features = 15)
vip(bst.mod.tag, num_features = 15)

tag_vocab_list <- c("text", "book", "person", "screenshot", "cartoon", "indoor", "outdoor", "danc", "cloth", "humanfac", "abstract", "design", "choreographi", "selfi", "woman")

# Retrieve tag_names from all data
tag_all <- projects %>%
  select(id, tag_names)

# Convert the blurb of entire dataset (both labeled and unlabeled) into sparse matrix
it_tag_all = itoken(tag_all$tag_names,
                    preprocessor = prep,
                    tokenizer = tag_tok,
                    ids = tag_all$id,
                    progressbar = FALSE)

# Vectorize all blurb
dtm_tag_all = create_dtm(it_tag_all, tag_vectorizer)
dtm_tag_all_bin <- dtm_tag_all>0+0

# Select out those terms and convert them into a dataframe so it can be bind with the original dataset
# Modify the variable names
dtm_tag_all_small <- dtm_tag_all_bin[, tag_vocab_list]
tag_dense = as.matrix(dtm_tag_all_small)+0
tag_vocab_df = as.data.frame(tag_dense)
tag_vocab_df <- tag_vocab_df %>%
  rename_with( ~ paste0("tag_contains_", .x))

# Replace 0 with NO, 1 with YES
tag_vocab_df[tag_vocab_df == 0] <- "NO"
tag_vocab_df[tag_vocab_df == 1] <- "YES"

#==============================features selection and cleaning===================================

# Select out the features that we are not using
projects_clean <- projects %>%
  select(-c(creator_id, name, creator_firstname, blurb, location_slug, category_name, minage_project, minage_creator, maxage_project, maxage_creator, accent_color, captions, tag_names, reward_amounts, reward_descriptions, location_place, category_freq, location_type_freq, goal_quantile_rank))


# Convert all tag_names and blurb vocabulary variables into factors
tag_col_names <- names(tag_vocab_df)
tag_vocab_df[,tag_col_names] <- lapply(tag_vocab_df[,tag_col_names], as.factor)

blurb_col_names <- names(blurb_vocab_df)
blurb_vocab_df[,blurb_col_names] <- lapply(blurb_vocab_df[,blurb_col_names], as.factor)

# Create a df before converting variables into dummies --> for ridge and lasso
no_dummy <- cbind(projects_clean, blurb_vocab_df)
no_dummy <- cbind(no_dummy, tag_vocab_df)

# Convert the factor variables (from tag_names and blurb) into dummy variable
tag_vocab_dummy <- dummyVars(~., data = tag_vocab_df)
one_hot_tag_vocab <- data.frame(predict(tag_vocab_dummy, newdata = tag_vocab_df))

blurb_vocab_dummy <- dummyVars(~., data = blurb_vocab_df)
one_hot_blurb_vocab <- data.frame(predict(blurb_vocab_dummy, newdata = blurb_vocab_df))

# Create dummy variables for other variables
dummy <- dummyVars( ~ . , data = projects_clean)
one_hot_projects <- data.frame(predict(dummy, newdata = projects_clean))

# Bind the vocab dummies with the original dummy variables data
one_hot_projects <- cbind(one_hot_projects, one_hot_blurb_vocab)
one_hot_projects <- cbind(one_hot_projects, one_hot_tag_vocab)

# Separate into training and testing again
labeled_data <- one_hot_projects %>%
  filter(is_train == 1)

unlabeled_data <- one_hot_projects %>%
  filter(is_train == 0)

#join the training y to the labeled data
#also turn two of the target variables into factors
labeled_data <- labeled_data %>%
  left_join(train_y, by = "id") %>%
  mutate(success = ifelse(success == "NO", 0, 1),
         success = as.factor(success)) %>%
  mutate(big_hit = ifelse(big_hit == "NO", 0, 1),
         big_hit = as.factor(big_hit))

# Drop one level of factor variables and the target variables that we are not using
labeled_data = labeled_data %>%
  select(-c(big_hit, backers_count, id, deadline_month.01, deadline_date.Early, create_month.01, create_date.Early, launch_month.01, launch_date.Early, location_type.County, region.ENCentral, category_parent.art, isbwImg1.NO, color_foreground.Black, color_background.Black, 
            isTextPic.NO, isLogoPic.NO, isCalendarPic.NO, isDiagramPic.NO, isShapePic.NO, contains_youtube.NO, is_train, launch_on_weekend.NO, location_state.AK, creator_female_name.F, big_project.NO, is_popular_cat.NO, minage_project_group.Elder, minage_creator_group.Elder, 
            maxage_project_group.Elder, maxage_creator_group.Elder, descriptionPositive.NEUTRAL, blurb_contains_album.NO, blurb_contains_app.NO, blurb_contains_book.NO, blurb_contains_cd.NO, blurb_contains_danc.NO, blurb_contains_design.NO, blurb_contains_festiv.NO,
            blurb_contains_film.NO, blurb_contains_game.NO, blurb_contains_music.NO, blurb_contains_art.NO, blurb_contains_premier.NO, blurb_contains_stori.NO, blurb_contains_theater.NO, blurb_contains_theatr.NO, tag_contains_abstract.NO, tag_contains_book.NO, 
            tag_contains_cartoon.NO, tag_contains_choreographi.NO, tag_contains_cloth.NO, tag_contains_danc.NO, tag_contains_design.NO, tag_contains_humanfac.NO, tag_contains_indoor.NO, tag_contains_outdoor.NO, tag_contains_person.NO, tag_contains_screenshot.NO, 
            tag_contains_selfi.NO, tag_contains_text.NO, tag_contains_woman.NO))

# Drop one level of factor variables in the testing/unlabeled data as well
unlabeled_data = unlabeled_data %>%
  select(-c(id, deadline_month.01, deadline_date.Early, create_month.01, create_date.Early, launch_month.01, launch_date.Early, location_type.County, region.ENCentral, category_parent.art, isbwImg1.NO, color_foreground.Black, color_background.Black, 
            isTextPic.NO, isLogoPic.NO, isCalendarPic.NO, isDiagramPic.NO, isShapePic.NO, contains_youtube.NO, is_train, launch_on_weekend.NO, location_state.AK, creator_female_name.F, big_project.NO, is_popular_cat.NO, minage_project_group.Elder, minage_creator_group.Elder, 
            maxage_project_group.Elder, maxage_creator_group.Elder, descriptionPositive.NEUTRAL, blurb_contains_album.NO, blurb_contains_app.NO, blurb_contains_book.NO, blurb_contains_cd.NO, blurb_contains_danc.NO, blurb_contains_design.NO, blurb_contains_festiv.NO,
            blurb_contains_film.NO, blurb_contains_game.NO, blurb_contains_music.NO, blurb_contains_art.NO, blurb_contains_premier.NO, blurb_contains_stori.NO, blurb_contains_theater.NO, blurb_contains_theatr.NO, tag_contains_abstract.NO, tag_contains_book.NO, 
            tag_contains_cartoon.NO, tag_contains_choreographi.NO, tag_contains_cloth.NO, tag_contains_danc.NO, tag_contains_design.NO, tag_contains_humanfac.NO, tag_contains_indoor.NO, tag_contains_outdoor.NO, tag_contains_person.NO, tag_contains_screenshot.NO, 
            tag_contains_selfi.NO, tag_contains_text.NO, tag_contains_woman.NO))


# Output the datasets accordingly for further training
write.csv(labeled_data, "labeled_data.csv", row.names = FALSE)
write.csv(unlabeled_data, "unlabeled_data.csv", row.names = FALSE)
write.csv(no_dummy, "no_dummy.csv", row.names = FALSE)
write.csv(train_y, "train_y.csv", row.names = FALSE)





# Output three dataset: 1) no external data and text mining 2) only with external data 3) with only text mining -----> for comparison
df_noexternal_notextmining <- labeled_data %>%
  select(-c(starts_with("creator_female_name"), starts_with("blurb_contains_"), starts_with("tag_contains")))

write.csv(df_noexternal_notextmining, "df_noexternal_notextmining.csv", row.names = FALSE)

df_onlyexternal <- labeled_data %>%
  select(-c(starts_with("blurb_contains_"), starts_with("tag_contains_")))

write.csv(df_onlyexternal, "df_onlyexternal.csv", row.names = FALSE)

df_onlytextmining <- labeled_data %>%
  select(-c(starts_with("creator_female_name")))

write.csv(df_onlytextmining, "df_onlytextmining.csv", row.names = FALSE)


