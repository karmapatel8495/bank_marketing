if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, mice, e1071, Metrics, skimr, pracma, shiny)

df <- read.csv("https://www.dropbox.com/s/lyi5bw2fxtuahm9/bank.csv?dl=1", sep = ";", stringsAsFactors = F)
skim(df)

chr <- df[,sapply(df,is.character)]
int <- df[,sapply(df,is.integer)]
chr[is.na(chr)] <- "Not Available"
fac <- chr %>% lapply(as.factor) %>% as.data.frame()
df <- bind_cols(fac,int)
micemod <- df %>% mice(method='rf')
df <- complete(micemod)

y <- df$y

rm(int, chr)

svm_model<-svm(y~., data=df, cost = 3)

input <- df[0,]

ui <- fluidPage(
  sliderInput(inputId = "age", label = "Age", value = mean(df$age), min = min(df$age), max = max(df$age)),
  selectInput(inputId = "job", label = "Job Type", choices = df$job, multiple = FALSE),
  selectInput(inputId = "marital", label = "Marital Status", choices = df$marital, multiple = FALSE),
  selectInput(inputId = "education", label = "Education Level", choices = df$education, multiple = FALSE),
  selectInput(inputId = "default", label = "Has Credit in Default?", choices = df$default, multiple = FALSE),
  numericInput(inputId = "balance", label = "Average yearly balance (in Euros):", value = 0),
  selectInput(inputId = "housing", label = "Has Housing Loan?", choices = df$housing, multiple = FALSE),
  selectInput(inputId = "loan", label = "Has Personal Loan?", choices = df$loan, multiple = FALSE),
  selectInput(inputId = "contact", label = "Contact communication type", choices = df$contact, multiple = FALSE),
  selectInput(inputId = "day", label = "Last contact day of the month", choices = df$day, multiple = FALSE),
  selectInput(inputId = "month", label = "Last contact month of year", choices = df$month, multiple = FALSE),
  sliderInput(inputId = "duration", label = "Last contact duration (in seconds)", value = mean(df$duration), min = min(df$duration), max = max(df$duration)),
  numericInput(inputId = "campaign", label = "Number of contacts performed during this campaign and for this client?", value = 0),
  numericInput(inputId = "pdays", label = "Number of days that passed by after the client was last contacted from a previous campaign:", value = 0),
  numericInput(inputId = "previous", label = "Number of contacts performed before this campaign and for this client:", value = 0)
)



server <- function(input, output){}


shinyApp(ui = ui, server = server)
