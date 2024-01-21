# Dr Clippy

## Overview
This is a simple toy that replaces acts as an aid to the conversation chain between a decision maker and a data analyst. The idea is that you can ask Dr Clippy to do some simple classification modeling on top of voter data and it will:

* Generate a query to pull the necessary data from BigQuery (LangChain + OpenAI)
* Prep the data for modeling (scikit-learn)
* Train the model (scikit-learn)
* Analyze the results and provide some feedback on to how well you did with feature selection (LangChain + OpenAI)

All in all it is relatively rudimentary but it does have my intended effect which is to show how AI can augment the feedback loop between two human actors. This does not relace an analyst, but it allows a relatively non-technical person to iterate on their ideas before pulling someone else in. 

## Setup
The biggest thing you'll need to do is dump your database schema into a JSON file for ingestion. Anything too wild and you'll hit token limits for OpenAI so be selective with the tables and fields you include. Descriptions help the LLM determine what it can answer and how to formulate your query.

Other than that, it's a dead standard Python + `pipenv` script that relies on the Google API so don't forget your `gcloud auth application-default login`.

## Author
You can find me on Twitter @matthewstafford or email me at matthew.stafford@gmail.com.

