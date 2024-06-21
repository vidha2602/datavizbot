import openai
import json
import ast
import os
import chainlit as cl
import requests
import os
import time
import io
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import tiktoken
import langchain
import openai
import os
from langchain.tools.python.tool import PythonAstREPLTool
import time
# import dask.dataframe as dd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional
import re
from datetime import datetime, timedelta

import pandas as pd
openai.api_key = "YOUR_OPEN_API_KEY"
os.environ['OPENAI_API_KEY']=openai.api_key

file_name = "AdidasSalesdata.xlsx"
print("Reading file...: ", file_name)
df = pd.read_excel(file_name)#, dtype={col: 'object' for col in pd.read_csv('Tro_sales_867_data.csv', nrows=1).columns})
#df = pd.DataFrame({"ID": [1,2,3], "Name": ["a", "b", "c"]})
print("Reading done!")

MAX_ITER = 15

def ask_gpt_3_turbo_model_4k(user_question_content, system_content = "You are a helpfull assistant"):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",#-16k",
      messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(user_question_content)},],
        temperature = 0,
        max_tokens = 1000,
    )
    output_response = str(response["choices"][0]["message"]["content"])
    return output_response

python_repl = PythonAstREPLTool()
def repl_tool_fun(code):
    print("code input in repl_tool_fun: ", )
    print(code)
    output= python_repl.run(code)
    print("output: output: ", output)
    return str(output)
print("loading file into python env at 130 line...., i.e loading code is running.....")
loading_code = """
import pandas as pd \n
import matplotlib as plt \n
import seaborn as sns \n
file_name = 'AdidasSalesdata.xlsx' \n
data = pd.read_excel('AdidasSalesdata.xlsx')\n
num_rows = data.shape[0] \n
print('num_rows: ', num_rows)"""
print(repl_tool_fun(loading_code))


functions = [
  {'name': 'run_python_code',
 'description': 'Useful you need to run python code for doing data analysis and plotting graphs',
            "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "python code to run to do required analysis",
                },
            },
            "required": ["query"]
        },
    },

    {


def extract_images_names(python_code):
    matches = re.findall(r"plt\.savefig\('(.*?)'", str(python_code))

    # Extract the image names if found
    image_names_lst = []
    if matches:
        for i, image_name in enumerate(matches):
            print(f"Saved image name {i+1}: {image_name}")
            image_names_lst.append(image_name)
    else:
        print("Image names not found")

    return image_names_lst

def extract_excel_names(python_code):
    matches = re.findall(r"(?:wb\.save|to_excel|to_csv)\('([^']+)'", str(python_code))

    # Extract the image names if found
    excel_names_lst = []
    image_names_lst = []
    if matches:
        for i, excel_name in enumerate(matches):
            print(f"Saved excel name {i+1}: {excel_name}")
            excel_names_lst.append(excel_name)
    else:
        print("Excel names not found")

    return excel_names_lst


@cl.step
async def function_output_streaming(new_delta, openai_message, content_ui_message, function_ui_message):
    if "name" in new_delta["function_call"]:
        openai_message["function_call"] = {
            "name": new_delta["function_call"]["name"]}
        await content_ui_message.send()
        function_ui_message = cl.Message(
            author=new_delta["function_call"]["name"],
            content="")#, indent=1, language="json")
        await function_ui_message.stream_token(new_delta["function_call"]["name"])

    if "arguments" in new_delta["function_call"]:
        if "arguments" not in openai_message["function_call"]:
            openai_message["function_call"]["arguments"] = ""
        openai_message["function_call"]["arguments"] += new_delta["function_call"]["arguments"]
        await function_ui_message.stream_token(new_delta["function_call"]["arguments"])
#    return openai_message, content_ui_message, function_ui_message


async def process_new_delta(new_delta, openai_message, content_ui_message, function_ui_message):
    #content_ui_message.disable_feedback =  False
    if "role" in new_delta:
        openai_message["role"] = new_delta["role"]
    if "content" in new_delta:
        new_content = new_delta.get("content") or ""
        openai_message["content"] += new_content
        await content_ui_message.stream_token(new_content)
    if "function_call" in new_delta:
        #await function_output_streaming(new_delta, openai_message, content_ui_message, function_ui_message)
        # async with cl.Step(name="gpt4", type="llm", root=False, show_input= False) as function_ui_message:
        #print("function_ui_message type test:", function_ui_message)
        if "name" in new_delta["function_call"]:
            openai_message["function_call"] = {
                "name": new_delta["function_call"]["name"]}
            await content_ui_message.send()
            # function_ui_message = cl.Message(
            #     author=new_delta["function_call"]["name"],
            #     content="")#, indent=1, language="json")
            # await function_ui_message.stream_token(new_delta["function_call"]["name"])

        if "arguments" in new_delta["function_call"]:
            if "arguments" not in openai_message["function_call"]:
                openai_message["function_call"]["arguments"] = ""
            openai_message["function_call"]["arguments"] += new_delta["function_call"]["arguments"]
            # await function_ui_message.stream_token(new_delta["function_call"]["arguments"])
    return openai_message, content_ui_message, function_ui_message





@cl.on_chat_start
async def start_chat():

    await cl.Message(content="Trust but verify each step of the bot to ensure that your question is interpreted and analyzed correctly.\n Choose from the following questions or type your own.",disable_feedback=True).send()


    actions_sales = [
    cl.Action(name="total_revenue", value="total_revenue", label="1. What is the total revenue generated over the given period?"),
    cl.Action(name="time_periods", value="time_periods", label="2. How do the total units sold compare across different time periods (days, weeks, months)"),
    cl.Action(name="region", value="region", label="3. Which region generates the highest revenue?"),

    ]
    await cl.Message(content="**Sales Performance Analysis:**", actions=actions_sales,disable_feedback=True).send()


    actions_salesbyregion = [
    cl.Action(name="product_categories", value="product_categories", label="1. Which product categories (Apparel, Street Footwear, Athletic Footwear) are the best sellers?"),
    cl.Action(name="gender_wise", value="gender_wise", label="2. What are the sales trends for men's and women's products?"),
    cl.Action(name="city_sales", value="city_sales", label="3. Which cities have the highest sales?"),
    ]
    await cl.Message(content="**Product Category Analysis:**", actions=actions_Trodelvy_trends,disable_feedback=True).send()


    # df = cl.user_session.get("df")
    # column_names = list(df.columns)
    print("inside start_Chart/............")
    column_names = """
    Below are the column names. Few columns given with some relevant information about that column for better context.
    'Retailer': The name of the retailer.
    'Retailer ID': A unique identifier for the retailer.
'Invoice Date': The date of the invoice (transaction date).
'Region': The geographical region where the sale occurred.
'State': The state where the sale occurred.
'City': The city where the sale occurred.
'Gender': The gender category of the product (e.g., Men, Women).
'Type': The type of product sold.
'Product Category': The category of the product (e.g., Apparel, Street Footwear).
'Price per Unit': The price of a single unit of the product.
'Units Sold': The number of units sold.
'Total Sales': The total sales amount (Price per Unit * Units Sold).
'Operating Profit': The profit made from the sale after operating expenses.
'Operating Margin': The operating profit as a percentage of total sales.
'Sales Method': The method through which the sale was made (e.g., Outlet).
    """#df.columns
    # column_names =  ['Covered_Recipient_Type', 'Teaching_Hospital_Name', 'Covered_Recipient_NPI', 'Covered_Recipient_First_Name', 'Covered_Recipient_Middle_Name', 'Covered_Recipient_Last_Name', 'Covered_Recipient_Name_Suffix', 'Recipient_Primary_Business_Street_Address_Line1', 'Recipient_Primary_Business_Street_Address_Line2', 'Recipient_City', 'Recipient_State', 'Recipient_Zip_Code', 'Recipient_Country', 'Covered_Recipient_Specialty_1', 'Covered_Recipient_License_State_code1', 'Principal_Investigator_1_Covered_Recipient_Type', 'Principal_Investigator_1_Profile_ID', 'Principal_Investigator_1_NPI', 'Principal_Investigator_1_First_Name', 'Principal_Investigator_1_Middle_Name', 'Principal_Investigator_1_Last_Name', 'Principal_Investigator_1_Name_Suffix', 'Principal_Investigator_1_Business_Street_Address_Line1', 'Principal_Investigator_1_Business_Street_Address_Line2', 'Principal_Investigator_1_City', 'Principal_Investigator_1_State', 'Principal_Investigator_1_Zip_Code', 'Principal_Investigator_1_Country', 'Principal_Investigator_1_Province', 'Principal_Investigator_1_Postal_Code', 'Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID', 'Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name', 'Related_Product_Indicator', 'Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_1', 'Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1', 'Total_Amount_of_Payment_USDollars', 'Date_of_Payment', 'Month_of_Payment', 'Form_of_Payment_or_Transfer_of_Value', 'Expenditure_Category1', 'Expenditure_Category2', 'Program_Year', 'Payment_Publication_Date', 'ClinicalTrials_Gov_Identifier', 'Research_Information_Link']
#list(df.columns)
    cl.user_session.set("df", df)
    # cl.user_session.set("file_name", file_name)
    file_name = "Tro_sales_867_data.csv"#cl.user_session.get("file_name")#file.name
    # print("Running loading python code .................")
    # and first 2 rows of the uploaded excel file are(i.e result of code df.head(2).to_string(index = False)) - \n {df.head(2).to_string(index = False)}
    inital_prompt = f"""You are a Data Analyst skilled in interpreting user queries and visualizing compelling insights using Pandas dataframes. You will write Python code and execute them to get the final answer to user questions. Follow the below steps while you answer the question:\n

   Step-1: First step is to divide the User question into multiple sub-tasks to help get accurate final answers, and Sub-tasks should simplify the task.
Step-2: Now, Perform each subtask by writing Python code and executing it using the `run_python_code` function. First, explain how you will perform this subtask via text, then write Python codes by applying appropriate operations on different columns of the given pandas dataframe. The operations can include - selecting the right columns of data, applying the right filters on the right selected columns with the right values. If the answer requires any chart, mention how you draw that chart. When you draw a chart, Save the dataframe table used for drawing the chart as an Excel or CSV file and Save the chart as an image (.png).
- Always Download the Output in a CSV/Excel file and Try drawing charts for most questions.
Note: The data table represents sales of Adidas products, so there is no need to filter for product names in any column.
Column names of the Pandas table are - ['Retailer', 'Retailer ID', 'Invoice Date', 'Region', 'State', 'City', 'Gender', 'Product Category', 'Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin', 'Sales Method'].
Instructions to follow while writing Python code:
1) The Python variables are persisted across runs so every time you don't need to generate code for the tool from scratch that loads the data and other variables instead, you can use those variables directly in the code. Currently, the table loaded into a pandas dataframe variable called "sales_data" so you can use it.
2) First explain what you are going to do and how you are planning to give an answer to the current user question in one to two lines.
3) If you are not clear which column to use, you can select the best possible column in the table by observing all the column names in the table.
4) The generated Python code should have a specific focus on case-insensitive data handling. This means that when comparing the data in a given question and the data in the data frame columns, our code should perform case-insensitive checks (i.e., `str.lower()` keyword). Moreover, your code should implement partial matching of values by utilizing `str.contains()` methods instead of an exact match ('=').
5) Also use '.*' regular expression to match variations of strings (e.g., "foot locker" with "Foot Locker") when matching for store or product names.
6) You can perform one operation at a time.
7) The generated python code should include code to plot charts such as bar chart, line chart, pie chart, etc., wherever possible and should save all the plots with the name 'downloaded_image.png' only.
8) If the code creates any kind of plots using matplotlib/seaborn, the code must also have a line like "plt.savefig('downloaded_image.png', dpi=300, bbox_inches='tight')" that downloads the image of the plot. Do not include `plt.show()` in the code.
9) Generate the code lines in new lines.
10) While writing Python code, avoid using double quotes within your code; use single quotes instead, as the code is stored as a dictionary value and retrieved later.
11) If any error occurs when executing the Python code generated, first find reasons for that error and explain how you are going to fix it and why this should work. Then re-write the code to get the final answer.
    \n
    - Save outputs into a CSV/Excel if the output is a table or a chart. Charts are good way to explain the answer for analytical questions, include code for creating charts and saving them. Don't give download links in your final answer. \n
    Business Rules:
- For any analysis involving weekly sales, use the 'Invoice Date' column.
- Use the 'Retailer' column to analyze sales by retailer, 'Region' for sales by region, 'State' for sales by state, 'City' for sales by city, 'Gender' for gender-based analysis, and 'Product Category' for product category analysis.
- Consider 'Retailer ID' as the unique identifier for retailers in the data.
    """
    initial_code = """
import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
file_path = '/mnt/data/AdidasSalesdata.xlsx'
sales_data = pd.read_excel(file_path)

# Convert 'Invoice Date' to datetime
sales_data['Invoice Date'] = pd.to_datetime(sales_data['Invoice Date'])

# Aggregate total sales by week
sales_data['Week'] = sales_data['Invoice Date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_sales = sales_data.groupby('Week')['Total Sales'].sum().reset_index()

# Plot weekly sales
plt.figure(figsize=(12, 6))
plt.plot(weekly_sales['Week'], weekly_sales['Total Sales'], marker='o', linestyle='-')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.title('Weekly Sales Trend')
plt.grid(True)
plt.savefig('downloaded_image.png', dpi=300, bbox_inches='tight')

print(weekly_sales.head(10))
"""

    cl.user_session.set(
        "message_history",
        [{"role": "system","content":initial_promot}],
        )

@cl.action_callback("total_revenue")
async def on_top_decliners(action: cl.Action):
    await cl.Message(content = "What is the total revenue generated over the given period?",author="You").send()
    await handle_user_query("What is the total revenue generated over the given period?")

@cl.action_callback("time_periods")
async def on_growth_percentage(action: cl.Action):
    await cl.Message(content = "How do the total units sold compare across different time periods (days, weeks, months.",author="You").send()
    await handle_user_query("How do the total units sold compare across different time periods (days, weeks, months.")

@cl.action_callback("region")
async def on_growth_percentage(action: cl.Action):
    await cl.Message(content = "Which region generates the highest revenue?",author="You").send()
    await handle_user_query("Which region generates the highest revenue?")

@cl.action_callback("product_categories")
async def on_growth_percentage(action: cl.Action):
    await cl.Message(content = "Which product categories (Apparel, Street Footwear, Athletic Footwear) are the best sellers?",author="You").send()
    await handle_user_query("Which product categories (Apparel, Street Footwear, Athletic Footwear) are the best sellers?")
    
    
@cl.action_callback("gender_wise")
async def on_top_decliners(action: cl.Action):
    await cl.Message(content = "What are the sales trends for men's and women's products?",author="You").send()
    await handle_user_query("What are the sales trends for men's and women's products?")
    

@cl.action_callback("city_sales")
async def on_top_growers(action: cl.Action):
    await cl.Message(content = "Calculate the top 10 parent growers based on the sales growth percentage in Recent 13 Weeks for High Parent BC segment.",author="You").send()
    await handle_user_query("Which cities have the highest sales?")


@cl.on_message
async def run_conversation(message: cl.Message):
    await handle_user_query(message.content)

async def handle_user_query(user_message):
    email = cl.user_session.get("user").identifier
    user_message_copy = user_message
    print("type:  ", type(cl.user_session))
    message_history = cl.user_session.get("message_history")
    print("message_history at start: ", message_history)

    for message in message_history:
        if message['role'] == 'function':
            message['content'] = ""

    print("after removing function content: ", message_history)

    if len(message_history) > 15:
        del message_history[5:len(message_history) - 9]

    message_history.append({"role": "user", "content": user_message})

    cur_iter = 0

    while cur_iter < MAX_ITER:

        # OpenAI call
        openai_message = {"role": "", "content": ""}
        function_ui_message = None
        content_ui_message = cl.Message(content="")
        async with cl.Step(name = "C3PO", type = "llm", root = True, disable_feedback = False) as content_ui_message:
            async with cl.Step(name="Python Code", type="llm", root=False, show_input= False, language = "python") as function_ui_message:
                async for stream_resp in await openai.ChatCompletion.acreate(
                    #model="gpt-3.5-turbo-0613",
                    model="gpt-4-turbo",#"gpt-4", #-1106-preview",#"gpt-3.5-turbo-16k", #"gpt-4-0613",
                    messages=message_history,
                    stream=True,
                    function_call="auto",
                    functions=functions,
                    temperature=0
                ):

                    new_delta = stream_resp.choices[0]["delta"]
                    openai_message, content_ui_message, function_ui_message = await process_new_delta(
                        new_delta, openai_message, content_ui_message, function_ui_message)

                message_history.append(openai_message)
            print("openai_message at 583: ", openai_message)
            # if function_ui_message is not None:
            #     await function_ui_message.send()

            if stream_resp.choices[0]["finish_reason"] == "stop":
                break

            elif stream_resp.choices[0]["finish_reason"] != "function_call":
                raise ValueError(stream_resp.choices[0]["finish_reason"])


            function_name = openai_message.get("function_call").get("name")
            argument_str = openai_message.get("function_call").get("arguments")

            # Preprocessing to handle potential multiline string issues
            # This is a simplistic approach and might need to be adjusted
            # based on the actual format of the string you receive.
            # argument_str = argument_str.replace("\n", "\\n").replace('"', '"""')
            # arguments = ast.literal_eval(argument_str)

            if function_name=="run_python_code" or  function_name == "run_python":
                arguments = ast.literal_eval(argument_str)
            else:
                arguments = ast.literal_eval(argument_str)

            await function_ui_message.stream_token(python_ui_message)
            if function_name=='run_python_code' or  function_name == "run_python":
                try:

                    function_response = repl_tool_fun(arguments.get("query"))
                except:
                    function_response = repl_tool_fun(arguments.get("query"))

            # if function_name=='create_deck':
            #     print("success")

            # elif function_name=='run_embedding_search':
            #     print("in embedding search!!!!!")
            #     function_response, context = get_unstructured_answer(arguments.get("query"))
            #     print("context at 625: ", context)
            #     await cl.Message(
            #         author=function_name,
            #         content="Sources: \n" +str(context),
            #         language="json",
            #     ).send()




#
            # else:
            #     print("function is not being called")



            message_history.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )



            # await cl.Message(
            #     author="Python Code Execution Output",
            #     content = "\n\nAnswer: \n"+str(function_response),
            #     language = "json",
            # ).send()
            if function_name == 'run_python_code':
                async with cl.Step(name="Python Code Execution Output", type="llm", root=False, show_input= False, language = "python") as python_code_exc_output:
                    python_code_exc_output.output = function_response
                python_code = arguments.get("query")
                images_files_list = extract_images_names(python_code)
                excel_files_list = extract_excel_names(python_code)

                ppt_files_list = extract_ppt_names(python_code)

                excel_files_list.extend(ppt_files_list)



            if function_name == 'run_python_code' and len(images_files_list)>0:
                print("images_files_list: ", images_files_list)
                for i, image_file in enumerate(images_files_list):
                    filename = image_file
                    if os.path.exists(filename):

                        time.sleep(1)
                        elements = [cl.Image(name="image_"+str(time.time()), display="inline",size = "large", path=f"./{filename}")]

                        await cl.Message(content=f"Chart:{i+1}", elements=elements).send()

                        time.sleep(1)
                        if os.path.exists(filename):
                            try:
                                os.remove(filename)
                                print(f"{filename} has been deleted.")
                            except Exception as e:
                                print(f"Error while deleting {filename}: {str(e)}")
                        else:
                            print(f"{filename} does not exist.")

            else:
                print("No images created for this question....")

            if function_name == 'run_python_code' and len(excel_files_list)>0:
                print("excel_files_list: ", excel_files_list)
                for i, excel_file in enumerate(excel_files_list):
                    filename = excel_file
                    if os.path.exists(filename):

                        time.sleep(1)
                        #elements = [cl.Image(name="image_"+str(time.time()), display="inline",size = "large", path=f"./{filename}")]
                        elements = [cl.File(name= filename, path=f"./{filename}",display="inline",),]

                        await cl.Message(content=f"File:{i+1}", elements=elements).send()

                        time.sleep(1)
                        try:
                            os.remove(filename)
                            print(f"{filename} has been deleted.")
                        except Exception as e:
                            print(f"Error while deleting {filename}: {str(e)}")

            else:
                print("No excel files have been created for this question....")




            cur_iter += 1

    content_ui_message.remove()
    print("message_history after question :", message_history)
    #cl.user_session.set("message_history", message_history)
