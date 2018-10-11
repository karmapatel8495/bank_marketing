library(readr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(pander)
library(knitr)
library(h2o)
h2o.init()

jobList <- c('admin',
             'blue-collar',
             'entrepreneur',
             'housemaid',
             'management',
             'retired',
             'self-employed',
             'services',
             'student',
             'technician',
             'unemployed',
             'unknown')

maritalList <- c('divorced','married','single','unknown')

educationList  <- c('basic.4y',
                    'basic.6y',
                    'basic.9y',
                    'high.school',
                    'illiterate',
                    'professional.course',
                    'university.degree',
                    'unknown')

yesnoList <- c('no',
               'yes',
               'unknown')

contactList <- c('cellular','telephone')

dayList <- c('Monday',
             'Tuesday',
             'Wednesday',
             'Thursday',
             'Friday')

monthList <- c('January',
               'February',
               'March',
               'April',
               'May',
               'June',
               'July',
               'August',
               'September',
               'October',
               'November',
               'December'
)

poutList <- c('Failure',
              'Non-existent',
              'Success')

gbm <- readRDS(gzcon(url("https://www.dropbox.com/s/yuzcejfzfi9xiiy/gbm.rds?dl=1")))
rf <- readRDS(gzcon(url("https://www.dropbox.com/s/zleo3q5ofiva4gy/rf.rds?dl=1")))
nn <- readRDS(gzcon(url("https://www.dropbox.com/s/q16mawn0irt70on/nn.rds?dl=1")))

ui <- fluidPage(
  tags$head(
    tags$style(HTML("hr {border-top: 2px solid #000000;}")),
    tags$link(rel = "stylesheet", type = "text/css", 
              href = "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.css")
  ),
  
  titlePanel("Prediction of marketing efficiency"),
  sidebarLayout(
    sidebarPanel(
      helpText("Get graphical summary of marketing data"),
      textInput(inputId = "age", 
                label = "Age", 
                width = "400px"),
      textInput(inputId = "campaign", 
                label = "Number of contacts performed during this campaign for this client", 
                width = "400px"),
      textInput(inputId = "previous", 
                label = "Number of contacts performed before this campaign for this client", 
                width = "400px"),
      selectInput(inputId = "job",
                  label = "Job Type",
                  choices = jobList,
                  selected = NULL,
                  multiple = FALSE),
      selectInput(inputId = "marital", 
                  label = "Marital Status",
                  choices = maritalList,
                  selected = NULL,
                  multiple = FALSE),
      selectInput(inputId = "education",
                  label = "Education Level",
                  choices = educationList,
                  selected = NULL,
                  multiple = FALSE),
      selectInput(inputId = "default",
                  label = "Has Credit in Default?",
                  choices = yesnoList,
                  selected = NULL,
                  multiple = FALSE),
      selectInput(inputId = "housing", 
                  label = "Has Housing Loan?",
                  choices = yesnoList, 
                  selected = NULL,
                  multiple = FALSE),
      selectInput(inputId = "loan",
                  label = "Has Personal Loan?",
                  choices = yesnoList,
                  selected = NULL,
                  multiple = FALSE),
      selectInput(inputId = "contact",
                  label = "Contact communication type",
                  choices = contactList,
                  selected = NULL,
                  multiple = FALSE),
      selectInput(inputId = "day",
                  label = "Last contact day of the month",
                  choices = dayList,
                  selected = NULL,
                  multiple = FALSE),
      selectInput(inputId = "month",
                  label = "Last contact month of year",
                  choices = monthList,
                  selected = NULL,
                  multiple = FALSE),
      selectInput(inputId = "pout",
                  label = "Outcome of the previous marketing campaign?",
                  choices = poutList, 
                  selected = NULL,
                  multiple = FALSE),
      checkboxInput("showgbm", "Show/Hide GBM Prediction", value = FALSE),
      checkboxInput("showrf", "Show/Hide Random Forest Prediction", value = FALSE),
      checkboxInput("shownn", "Show/Hide Neural Network Prediction", value = FALSE)
    ),
    mainPanel(
      tabsetPanel(type = "tabs",
                  tabPanel("Prediction", 
                           hr(),
                           textOutput("age"),
                           textOutput("previous"),
                           textOutput("job"),
                           textOutput("marital"),
                           textOutput("education"),
                           textOutput("default"),
                           textOutput("housing"),
                           textOutput("loan"),
                           textOutput("contact"),
                           textOutput("day"),
                           textOutput("month"),
                           textOutput("pout"),
                           textOutput("pred"), 
                           hr(),
                           hr(),
                           textOutput("predgbm"),
                           textOutput("predrf"),
                           textOutput("prednn")
                  ),
                  tabPanel("GBM Summary", verbatimTextOutput("gbm")),
                  tabPanel("Random Forest Summary", verbatimTextOutput("rf")),
                  tabPanel("Neural Network summary", verbatimTextOutput("nn"))
      )
    )
  )
)

server <- function(input, output, session){

  predictMarketing <- reactive({

    input_data <- data.frame(matrix(ncol = 54, nrow = 0))
    
    age_1 <- as.numeric(input$age < 30)
    age_2 <- as.numeric(input$age >= 30 & input$age <= 60)
    age_3 <- as.numeric(input$age > 60)
    
    job_1 <- as.numeric(input$job == "admin")
    job_2 <- as.numeric(input$job == "blue_collar")
    job_3 <- as.numeric(input$job == "entrepreneur")
    job_4 <- as.numeric(input$job == "housemaid")
    job_5 <- as.numeric(input$job == "management")
    job_6 <- as.numeric(input$job == "retired")
    job_7 <- as.numeric(input$job == "self-employed")
    job_8 <- as.numeric(input$job == "services")
    job_9 <- as.numeric(input$job == "student")
    job_10 <- as.numeric(input$job == "technician")
    job_11 <- as.numeric(input$job == "unemployed")
    
    marital_1 <- as.numeric(input$marital == "single")
    marital_2 <- as.numeric(input$marital == "married")
    marital_3 <- as.numeric(input$marital == "divorced")
    
    edu_1 <- as.numeric(input$education == "basic.4y")
    edu_2 <- as.numeric(input$education == "basic.6y")
    edu_3 <- as.numeric(input$education == "basic.9y")
    edu_4 <- as.numeric(input$education == "high.school")
    edu_5 <- as.numeric(input$education == "professional.course")
    edu_6 <- as.numeric(input$education == "university.degree")
    edu_7 <- as.numeric(input$education == "unknown")
    
    default_1 <- as.numeric(input$default == "no")
    default_2 <- as.numeric(input$default == "unknown")
    
    housing_1 <- as.numeric(input$housing == "no")
    housing_2 <- as.numeric(input$housing == "yes")
    
    loan_1 <- as.numeric(input$loan == "no")
    loan_2 <- as.numeric(input$loan == "yes")
    
    con_1 <- as.numeric(input$contact == "cellular")
    con_2 <- as.numeric(input$contact == "telephone")
    
    mar <- as.numeric(input$month == "mar")
    apr <- as.numeric(input$month  == "apr")
    may <- as.numeric(input$month  == "may")
    jun <- as.numeric(input$month  == "jun")
    jul <- as.numeric(input$month  == "jul")
    aug <- as.numeric(input$month  == "aug")
    sep <- as.numeric(input$month  == "sep")
    oct <- as.numeric(input$month  == "oct")
    nov <- as.numeric(input$month  == "nov")
    dec <- as.numeric(input$month  == "dec")
    
    mon <- as.numeric(input$day == "mon")
    tue <- as.numeric(input$day  == "tue")
    wed <- as.numeric(input$day  == "wed")
    thu <- as.numeric(input$day  == "thu")
    fri <- as.numeric(input$day  == "fri")
    
    poutcome1  <- as.numeric(input$pout == "failure")
    poutcome2  <- as.numeric(input$pout == "nonexistent")
    poutcome3  <- as.numeric(input$pout == "success")
    
    cols <- c("age", "campaign", "previous", "lnage", "age_1", "age_2", "age_3",
              "job_1", "job_2", "job_3", "job_4", "job_5", "job_6", "job_7",  "job_8", 
              "job_9", "job_10", "job_11", "marital_1", "marital_2", "marital_3", 
              "edu_1", "edu_2", "edu_3", "edu_4", "edu_5", "edu_6", "edu_7", "default_1", 
              "default_2", "housing_1", "housing_2", "loan_1", "loan_2", "con_1", "con_2",
              "mar", "apr", "may", "jun", "jul",  "aug", "sep", "oct","nov", "dec", "mon", 
              "tue", "wed", "thu", "fri", "poutcome1", "poutcome2", "poutcome3")
    
    colnames(input_data) <- cols
    
    current_obs <- c(as.numeric(input$age), as.numeric(input$campaign), as.numeric(input$previous), 
                     log(as.numeric(input$age)), age_1, age_2, age_3,
              job_1, job_2, job_3, job_4, job_5, job_6, job_7,  job_8, 
              job_9, job_10, job_11, marital_1, marital_2, marital_3, 
              edu_1, edu_2, edu_3, edu_4, edu_5, edu_6, edu_7, default_1, 
              default_2, housing_1, housing_2, loan_1, loan_2, con_1, con_2,
              mar, apr, may, jun, jul,  aug, sep, oct, nov, dec, mon, 
              tue, wed, thu, fri, poutcome1, poutcome2, poutcome3)
    
    colnames(input_data) <- cols
    input_data[1,] <- current_obs
    
    h2o.no_progress()
    
    # GBM
    if (input$showgbm) {
      prediction_gbm <-  h2o.predict(gbm, newdata = as.h2o(input_data))
      output_gbm <- as.data.frame(prediction_gbm)
      output_value_gbm <- output_gbm$predict[1]
      output_no_accuracy_gbm <- round(as.numeric(output_gbm$no[1]) * 100, 4)
      output_yes_accuracy_gbm <- round(as.numeric(output_gbm$yes[1]) * 100, 4)
      if (output_value_gbm == "no") {
        output$predgbm <- renderPrint(cat(paste("Our GBM model predicts with ", output_no_accuracy_gbm,
                     "% accuracy that the client WON'T subscribe to a term deposit.")))
      }
      else {
        output$predgbm <- renderPrint(cat(paste("Our GBM model predicts with ", output_yes_accuracy_gbm,
                     "% accuracy that the client WOULD subscribe to a term deposit.")))
      }
    }
    
    # RF
    if (input$showrf) {
      prediction_rf <- h2o.predict(rf, newdata = as.h2o(input_data))
      output_rf <- as.data.frame(prediction_rf)
      output_value_rf <- output_rf$predict[1]
      output_no_accuracy_rf <- round(as.numeric(output_rf$no[1]) * 100, 4)
      output_yes_accuracy_rf <- round(as.numeric(output_rf$yes[1]) * 100, 4)
      if (output_value_rf == "no") {
        output$predrf <- renderPrint(cat(paste("Our Random  Forest model predicts with ", output_no_accuracy_rf,
                     "% accuracy that the client WON'T subscribe to a term deposit.")))
      }
      else {
        output$predrf <- renderPrint(cat(paste("Our Random Forest model predicts with ", output_yes_accuracy_rf,
                     "% accuracy that the client WOULD subscribe to a term deposit.")))
      }
    }
    
    # NN
    if (input$shownn) {
      prediction_nn <- h2o.predict(nn, newdata = as.h2o(input_data))
      output_nn <- as.data.frame(prediction_nn)
      output_value_nn <- output_nn$predict[1]
      output_no_accuracy_nn <- round(as.numeric(output_nn$no[1]) * 100, 4)
      output_yes_accuracy_nn <- round(as.numeric(output_nn$yes[1]) * 100, 4)
      if (output_value_nn == "no") {
        output$prednn <- renderPrint(cat(paste("Our Neural Network model predicts with ", output_no_accuracy_nn,
                     "% accuracy that the client WON'T subscribe to a term deposit.")))
      }
      else {
        output$prednn <- renderPrint(cat(paste("Our Neural  Network model predicts with ", output_yes_accuracy_nn,
                     "% accuracy that the client WOULD subscribe to a term deposit.")))
      }
    }
  })
  
  output$age <- renderText({ 
    paste("Age = ",input$age)
  })
  output$campaign <- renderText({ 
    paste("Number of contacts performed during this campaign for this client = ",input$campaign)
  })
  output$previous <- renderText({ 
    paste("Number of contacts performed before this campaign for this client = ",input$previous)
  })
  output$job <- renderText({ 
    paste("Job Type = ",input$job)
  })
  output$marital <- renderText({ 
    paste("Marital Status = ",input$marital)
  })
  output$education <- renderText({ 
    paste("Education Level = ",input$education)
  })
  output$default <- renderText({ 
    paste("Has Credit in Default? = ",input$default)
  })
  output$housing <- renderText({ 
    paste("Has Housing Loan? = ",input$housing)
  })
  output$loan <- renderText({ 
    paste("Has Personal Loan? = ",input$loan)
  })
  output$contact <- renderText({ 
    paste("Contact communication type = ",input$contact)
  })
  output$day <- renderText({ 
    paste("Last contact day of the month = ",input$day)
  })
  output$month <- renderText({ 
    paste("Last contact month of year = ",input$month)
  })
  output$pout <- renderText({ 
    paste("Outcome of the previous marketing campaign? = ",input$pout)
  })
  output$pred <- renderPrint(
      predictMarketing()
    )
  output$gbm <- renderPrint({ 
    summary(gbm)
  })
  output$rf <- renderPrint({ 
    summary(rf)
  })
  output$nn <- renderPrint({ 
    summary(nn)
  })
}

shinyApp(ui = ui, server = server)


