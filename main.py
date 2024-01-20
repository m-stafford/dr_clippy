import json
import openai
import langchain
import argparse
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from google.cloud import bigquery

def read_json(file_path):
    # We only read it into a dict to verify that it is valid JSON
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def generate_sql(user_input, json_data, prompt):
    prompt_template = PromptTemplate.from_template(prompt)

    model = ChatOpenAI(model_name="gpt-4-1106-preview")

    #chain = LLMChain(llm=llm, prompt=prompt_template)

    chain = prompt_template | model | StrOutputParser()
    sql_query = chain.invoke({"question": user_input, "schema": json.dumps(json_data, indent=2)})

    return sql_query

def analyze_results(results, prompt):
    prompt_template = PromptTemplate.from_template(prompt)

    model = ChatOpenAI(model_name="gpt-4-1106-preview")

    #chain = LLMChain(llm=llm, prompt=prompt_template)

    chain = prompt_template | model | StrOutputParser()
    results = chain.invoke({"results": json.dumps(results, indent=2)})


    return results

def execute_bigquery_query(query):
    client = bigquery.Client()
    query_job = client.query(query)
    results = query_job.result()
    data = results.to_dataframe()
    return data

def perform_classification(data):
    X = data.drop('target_variable', axis=1)

    def custom_combiner(feature, category):
        return str(feature) + "_" + type(category).__name__ + "_" + str(category)

    ohe = OneHotEncoder(feature_name_combiner=custom_combiner)
    X = ohe.fit_transform(X)

    y = data['target_variable']

    model = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    model_info = lr_coef_to_json(model.coef_, model.classes_, ohe.get_feature_names_out())
    model_info["accuracy"] = accuracy
    model_info["number_of_classes"] = len(model.classes_)

    #print(json.dumps(model_info, indent=2))
    return model_info

def lr_coef_to_json(coef, classes, feature_names):
    # Thank you ChatGPT for writing the kinds of functions I hate writing
    """
    Converts logistic regression coefficients and classes into a descriptive JSON object.

    :param coef: Coefficients of the logistic regression model
    :param classes: Class labels of the logistic regression model
    :param feature_names: Names of the features
    :return: JSON object with class labels as keys and coefficient dictionaries as values
    """
    # Ensure the number of classes and the shape of coef array match
    if len(classes) != len(coef):
        raise ValueError("Length of classes should match the number of rows in coef")

    # Ensure the number of features matches the number of coefficients in each class
    if len(feature_names) != len(coef[0]):
        raise ValueError("Number of feature names should match the number of coefficients per class")

    # Create a dictionary for the JSON object
    class_coef_dict = {}

    for class_index, class_label in enumerate(classes):
        # Map each feature name to its corresponding coefficient
        feature_coef_dict = {feature: coef[class_index][i] for i, feature in enumerate(feature_names)}
        class_coef_dict[class_label] = feature_coef_dict

    return class_coef_dict

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_string', help='')
    args = parser.parse_args()

    json_data = read_json('tables.json')

    with open("query_prompt.txt", "r") as query_prompt_file:
        query_prompt = query_prompt_file.read()

        print(f"So it looks like you're trying to analyze some data and want to know '{args.input_string}'. I can help you with that!\n")
        sql_query = generate_sql(args.input_string, json_data, query_prompt)

    print(f"To get the data we need, I'm going to run the following query: \n {sql_query}\n")
    query_results = execute_bigquery_query(sql_query)

    classification_result = perform_classification(query_results)

    with open("analysis_prompt.txt", "r") as analysis_prompt_file:
        analysis_prompt = analysis_prompt_file.read()
        analysis = analyze_results(classification_result, analysis_prompt)
        print(f"Okay, I've gone ahead and analyized your results! \n {analysis}")


if __name__ == "__main__":
    main()

