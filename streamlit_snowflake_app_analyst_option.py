# Import necessary modules
import os
import json
import streamlit as st
import pandas as pd
import networkx as nx
from graphviz import Digraph
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
import warnings
import _snowflake
from typing import List, Dict, Optional

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Get the active Snowpark session
session = Session.builder.getOrCreate()

# Configuration variables for the semantic search
STAGE = "cortex_stage"
FILE = "simple_semantic_model.yaml"

# Functions for the lineage visualization app

@st.cache_data
def fetch_lineage_data():
    try:
        # Fetch the data from Snowflake using Snowpark session
        df = session.table("COLUMN_LINEAGE_MERGED").to_pandas()
    except SnowparkSQLException as e:
        st.error(f"Error fetching lineage data: {str(e)}")
        return pd.DataFrame()
    return df

def create_node_id(node_type, name, database='', table=''):
    # Create a unique node ID based on the entity
    if node_type == 'Database Column':
        # Include database and table name
        return f"DBCol_{database}_{table}_{name}".replace(" ", "_").replace(":", "_").replace(".", "_")
    elif node_type == 'Tableau Field':
        # Use the field name as the node ID to represent it as a single node across all sheets
        return f"Field_{name}".replace(" ", "_").replace(":", "_").replace(".", "_")
    elif node_type == 'Sheet':
        # Use the sheet name as the node ID
        return f"Sheet_{name}".replace(" ", "_").replace(":", "_").replace(".", "_")
    else:
        return f"{node_type}_{name}".replace(" ", "_").replace(":", "_").replace(".", "_")

def create_node_label(node_type, name, database='', table='', dashboard=''):
    if node_type == 'Database Column':
        # Display only the column name, with table name in brackets
        if table:
            return f"{name}\n({table})"
        else:
            return name
    elif node_type == 'Tableau Field':
        return name  # For fields, use the field name
    elif node_type == 'Sheet':
        if dashboard:
            return f"Sheet: {name}\n({dashboard})"
        else:
            return f"Sheet: {name}"
    else:
        return name  # For other types, use the name

def create_node_tooltip(node_type, name, database='', table='', transformations=None):
    # Create tooltip text with structured metadata
    tooltip = f"Type: {node_type}\n"
    if node_type == 'Database Column':
        tooltip += f"Database: {database}\n"
        tooltip += f"Table: {table}\n"
        tooltip += f"Column: {name}\n"
    elif node_type == 'Tableau Field':
        tooltip += f"Field: {name}\n"
    elif node_type == 'Sheet':
        tooltip += f"Sheet: {name}\n"
    else:
        tooltip += f"Name: {name}\n"
    if transformations:
        tooltip += "Transformations:\n" + "\n".join(transformations)
    else:
        tooltip += "Transformation: One-to-one mapping"
    return tooltip

def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        source_type = row['SOURCE_TYPE']
        target_type = row['TARGET_TYPE']
        source_name = row['SOURCE_NAME']
        target_name = row['TARGET_NAME']
        source_database = row.get('SOURCE_DATABASE', '')
        source_table = row.get('SOURCE_TABLE', '')
        target_database = row.get('TARGET_DATABASE', '')
        target_table = row.get('TARGET_TABLE', '')
        transformation = row.get('TRANSFORMATION', '')
        if not transformation:
            transformation = 'One-to-one mapping'
        sheet_name = row.get('SHEET', '')
        dashboard_name = row.get('DASHBOARD', '')  # Get the dashboard name

        # Create node IDs and labels
        source_node_id = create_node_id(
            source_type, source_name, source_database, source_table)
        source_label = create_node_label(
            source_type, source_name, source_database, source_table)

        if source_node_id not in G:
            G.add_node(
                source_node_id,
                label=source_label,
                type=source_type,
                name=source_name,
                database=source_database,
                table=source_table
            )
        else:
            # Ensure attributes are set even if the node already exists
            G.nodes[source_node_id].setdefault('name', source_name)
            G.nodes[source_node_id].setdefault('database', source_database)
            G.nodes[source_node_id].setdefault('table', source_table)

        # For target node
        target_node_id = create_node_id(
            target_type, target_name, target_database, target_table)
        target_label = create_node_label(
            target_type, target_name, target_database, target_table)

        if target_node_id not in G:
            G.add_node(
                target_node_id,
                label=target_label,
                type=target_type,
                name=target_name,
                database=target_database,
                table=target_table
            )
        # Ensure 'transformations' attribute exists
        G.nodes[target_node_id].setdefault('transformations', set())
        # Update transformations
        G.nodes[target_node_id]['transformations'].add(transformation)

        # Add edge from source to target
        G.add_edge(source_node_id, target_node_id)

        # If the target is a Tableau Field, connect it to the sheet
        if target_type == 'Tableau Field':
            # Sheet Node
            if sheet_name:
                sheet_node_id = create_node_id('Sheet', sheet_name)
                sheet_label = create_node_label('Sheet', sheet_name, dashboard=dashboard_name)
                if sheet_node_id not in G:
                    G.add_node(
                        sheet_node_id,
                        label=sheet_label,
                        type='Sheet',
                        name=sheet_name
                    )
                G.add_edge(target_node_id, sheet_node_id)
    return G

def get_lineage_subgraph(G, selected_node, max_upstream_depth=None, max_downstream_depth=None):
    # Collect upstream and downstream nodes up to the specified depth
    nodes_to_include = set()
    try:
        # Upstream nodes (ancestors)
        if max_upstream_depth is None:
            upstream_nodes = nx.ancestors(G, selected_node)
        else:
            reverse_G = G.reverse()
            lengths_up = nx.single_source_shortest_path_length(reverse_G, selected_node, cutoff=max_upstream_depth)
            upstream_nodes = set(lengths_up.keys()) - {selected_node}
        # Downstream nodes (descendants)
        if max_downstream_depth is None:
            downstream_nodes = nx.descendants(G, selected_node)
        else:
            lengths_down = nx.single_source_shortest_path_length(G, selected_node, cutoff=max_downstream_depth)
            downstream_nodes = set(lengths_down.keys()) - {selected_node}
        nodes_to_include.update(upstream_nodes)
        nodes_to_include.update(downstream_nodes)
        nodes_to_include.add(selected_node)
    except nx.NetworkXError as e:
        st.error(f"Error finding lineage for node {selected_node}: {e}")
        return None
    subgraph = G.subgraph(nodes_to_include)
    return subgraph

def plot_graph(subgraph, selected_nodes, G):
    if subgraph is None:
        st.write("No lineage information available.")
        return None

    dot = Digraph(comment='Lineage Graph', format='png')
    dot.attr('graph', rankdir='LR')  # Set layout to left-to-right

    for node_id in subgraph.nodes():
        node_data = G.nodes[node_id]
        node_type = node_data.get('type', '')
        node_label = node_data.get('label', node_id)
        name = node_data.get('name', '')
        database = node_data.get('database', '')
        table = node_data.get('table', '')
        transformations = node_data.get('transformations', set())
        node_tooltip = create_node_tooltip(node_type, name, database, table, transformations)
        node_style = 'filled'
        font_size = '10'

        if selected_nodes and node_id in selected_nodes:
            fillcolor = '#ffcccc'  # Light red
            color = 'red'
            shape = 'box'
        elif node_type == 'Sheet':
            shape = 'note'
            fillcolor = '#e0e0e0'  # Light grey
            color = 'black'
        elif node_type == 'Tableau Field':
            shape = 'ellipse'
            fillcolor = '#cce5ff'  # Light blue
            color = 'black'
        elif node_type == 'Database Column':
            shape = 'box'
            fillcolor = '#ccffcc'  # Light green
            color = 'black'
        else:
            shape = 'box'
            fillcolor = '#ffffff'
            color = 'black'

        # Set node attributes
        dot.node(node_id, label=node_label, shape=shape, style=node_style, fillcolor=fillcolor,
                 color=color, tooltip=node_tooltip, fontsize=font_size)

    for edge in subgraph.edges():
        dot.edge(edge[0], edge[1])

    return dot

def render_metadata_table(metadata_dict, display_names):
    """
    Render the metadata as a key-value table using Streamlit's native table rendering, 
    ensuring logical column names from `display_names`.
    """
    # Use display names for the metadata keys
    updated_metadata = {display_names.get(k, k): v for k, v in metadata_dict.items()}

    # Convert the updated metadata into a pandas DataFrame
    data = pd.DataFrame(
        list(updated_metadata.items()),  # Convert key-value pairs into a list of tuples
        columns=["Attribute", "Value"]  # Define column headers
    )

    # Replace empty or null values with 'One-to-one mapping'
    data["Value"] = data["Value"].replace(['', None], 'One-to-one mapping')

    # Display the metadata table using Streamlit's native table rendering
    st.table(data)

# Functions for the semantic search app (without changing any logic)

def send_message(prompt: str) -> Optional[dict]:
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "semantic_model_file": f"@{STAGE}/{FILE}",
    }

    try:
        response = _snowflake.send_snow_api_request(
            "POST", "/api/v2/cortex/analyst/message", {}, {}, request_body, {}, 30000
        )
        if response["status"] < 400:
            return json.loads(response["content"])
        else:
            st.warning(f"Semantic model lookup failed: {response}")
            return None
    except Exception as e:
        st.error(f"Error calling Cortex Analyst: {str(e)}")
        return None

def process_message(prompt: str) -> None:
    modified_prompt = f"""
You are an intelligent SQL assistant designed to generate accurate SQL queries based on user questions. Your task is to determine whether the question is directly related to a field name or column name (**Route 1**) or if it pertains to other entities like sheets, dashboards, or workbooks (**Route 2**). 

**Routing Logic:**

1. **Route 1 (Field/Column Related):**
   - **Criteria:** If the question explicitly mentions a specific field name or column name.
   - **Indicators:** Words like "field," "column," "data type," "details about," "contain," etc.
   - **Action:**
     - Use 'TARGET_NAME' for your search.
     - In your SQL query, select the columns: TARGET_TYPE, TARGET_NAME, TARGET_TABLE, TARGET_DATABASE, WORKBOOK, DASHBOARD, SHEET.
     - Utilize the `jarowinkler_similarity` function to find matches with a threshold of `0.95`.
     - **Ensure that the subquery returns a single row or handle multiple matches appropriately using `IN` or `EXISTS`.**

2. **Route 2 (Entity Related):**
   - **Criteria:** If the question is about entities such as sheets, dashboards, workbooks, or about the fields used within them.
   - **Indicators:** Phrases like "used to build," "included in," "fields of," "utilize," etc.
   - **Action:**
     - Identify the entity being inquired about (e.g., 'Location sheet').
     - Retrieve and return details specific to that entity, such as the list of fields used.
     - Use the `jarowinkler_similarity` function with an appropriate threshold to find the entity.
     - **Use the `IN` operator in the WHERE clause to handle cases where multiple matches are found.**

**Examples:**

- **Example 1: Route 1**
  - **Question:** "What is the data type of CUSTOMER_ID?"
  - **Route:** 1
  - **SQL Query:**
    ```sql
    SELECT TARGET_TYPE, TARGET_NAME, TARGET_TABLE, TARGET_DATABASE, WORKBOOK, DASHBOARD, SHEET
    FROM your_table
    WHERE jarowinkler_similarity(TARGET_NAME, 'CUSTOMER_ID') > 95;
    ```

- **Example 2: Route 1**
  - **Question:** "Show me details about the field ORDER_DATE."
  - **Route:** 1
  - **SQL Query:**
    ```sql
    SELECT TARGET_TYPE, TARGET_NAME, TARGET_TABLE, TARGET_DATABASE, WORKBOOK, DASHBOARD, SHEET
    FROM your_table
    WHERE jarowinkler_similarity(TARGET_NAME, 'ORDER_DATE') > 95;
    ```

- **Example 3: Route 2**
  - **Question:** "What fields are used to build the Location sheet?"
  - **Route:** 2
  - **SQL Query:**
    ```sql
    SELECT FIELD_NAME
    FROM your_table
    WHERE SHEET IN (
      SELECT SHEET
      FROM your_table
      WHERE jarowinkler_similarity(SHEET, 'Location sheet') > 95
    );
    ```

- **Example 4: Route 2**
  - **Question:** "Which fields are included in the Sales dashboard?"
  - **Route:** 2
  - **SQL Query:**
    ```sql
    SELECT FIELD_NAME
    FROM your_table
    WHERE DASHBOARD IN (
      SELECT DASHBOARD
      FROM your_table
      WHERE jarowinkler_similarity(DASHBOARD, 'Sales dashboard') > 95
    );
    ```

**Instructions:**

1. **Determine the Route:**
   - Analyze the question to decide whether it falls under **Route 1** or **Route 2** based on the criteria and indicators provided.

2. **Generate the SQL Query:**
   - For **Route 1**, construct the SQL query to retrieve details about the specified field or column.
   - For **Route 2**, construct the SQL query to retrieve details about the specified entity, such as the fields used in a particular sheet or dashboard.

3. **Handle Multiple Matches:**
   - **Use `IN` Instead of `=`**: When the subquery can return multiple rows, use the `IN` operator to handle all possible matches.
   - **Example:**
     ```sql
     WHERE SHEET IN (
       SELECT SHEET
       FROM your_table
       WHERE jarowinkler_similarity(SHEET, 'cust type') > 95
     )
     ```

4. **Use the SEARCH Function Appropriately:**
   - Apply the `jarowinkler_similarity` function to handle approximate string matching.
   - Ensure the similarity threshold is set appropriately (e.g., `> 95` for high similarity).

5. **Handle Edge Cases:**
   - If the question is ambiguous or doesn't clearly fit into either route, default to providing a clarification or asking for more information.

**Process the following question accordingly:**

'{prompt}'
"""


    # Send the message and get the response
    response = send_message(modified_prompt)

    if response:
        # Extract SQL query
        sql_query = None
        for item in response["message"]["content"]:
            if item["type"] == "sql":
                sql_query = item["statement"]
                break

        if sql_query:
            # Clean the SQL query (strip and remove trailing semicolons)
            cleaned_sql_query = sql_query.strip().rstrip(";")

            # Display the cleaned SQL query in a collapsed expander
            with st.expander("Show SQL Query", expanded=False):
                st.code(cleaned_sql_query, language="sql")

            # Run the SQL query and store results
            run_sql_and_store_results(cleaned_sql_query)
        else:
            st.error("No SQL query found in the response.")
    else:
        st.error("No semantic matches found for similar target_name entries.")

def run_sql_and_store_results(sql: str) -> None:
    try:
        # Execute the SQL query and get the result DataFrame
        result_df = session.sql(sql).to_pandas()

        if result_df.empty:
            st.error("The SQL query did not return any results.")
            return

        # Expected columns
        required_fields = [
            "TARGET_TYPE",
            "TARGET_NAME",
            "TARGET_TABLE",
            "TARGET_DATABASE",
            "WORKBOOK",
            "DASHBOARD",
            "SHEET"
        ]

        # Check if all required fields are present
        if all(field in result_df.columns for field in required_fields):
            # All required fields are present
            # Proceed as usual
            st.session_state['result_df'] = result_df
            st.session_state['route'] = 'route1'
        else:
            # Required fields are missing
            # Display the results and a message
            st.session_state['result_df'] = result_df
            st.session_state['route'] = 'route2'

    except Exception as e:
        st.error(f"SQL query execution failed: {str(e)}")

def reset_conversation() -> None:
    st.session_state.pop('result_df', None)  # Clear result_df
    st.session_state.pop('selected_option', None)  # Clear selected_option
    st.session_state.pop('user_question', None)  # Clear user question
    st.session_state.pop('route', None)
    # Clear selected filters
    st.session_state.pop('selected_target_type', None)
    st.session_state.pop('selected_workbook', None)
    st.session_state.pop('selected_dashboard', None)
    st.session_state.pop('selected_sheet', None)
    st.session_state.pop('selected_fields', None)
    st.session_state.pop('selected_table', None)
    st.session_state.pop('selected_columns', None)
    st.session_state.pop('target_database', None)

# Main Streamlit App

# Create columns for layout
col1, col2 = st.columns([3, 1])

def draw_legend():
    # Horizontal legend display
    st.markdown("""
        <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 10px;">
            <div style="margin-right: 20px;">
                <div style="width: 15px; height: 15px; background-color: #ffcccc;"></div>
                <p style="font-size: 10px; text-align: center;">Selected Node</p>
            </div>
            <div style="margin-right: 20px;">
                <div style="width: 15px; height: 15px; background-color: #ccffcc;"></div>
                <p style="font-size: 10px; text-align: center;">Database Column</p>
            </div>
            <div style="margin-right: 20px;">
                <div style="width: 15px; height: 15px; background-color: #cce5ff; border-radius: 50%;"></div>
                <p style="font-size: 10px; text-align: center;">Tableau Field</p>
            </div>
            <div style="margin-right: 20px;">
                <div style="width: 15px; height: 15px; background-color: #e0e0e0;"></div>
                <p style="font-size: 10px; text-align: center;">Sheet</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Define display names for metadata fields
metadata_display_names = {
    'WORKBOOK': 'Workbook Name',
    'DASHBOARD': 'Dashboard Name',
    'SHEET': 'Sheet Name',
    'TARGET_NAME': 'Field/Column Name',
    'TRANSFORMATION': 'Transformation',
    'SOURCE_TYPE': 'Source Type',
    'SOURCE_NAME': 'Source Name',
    'SOURCE_TABLE': 'Source Table',
    'SOURCE_DATABASE': 'Source Database',
    'TARGET_DATABASE': 'Target Database',
    'TARGET_TABLE': 'Target Table'
}

with col1:
    st.title('Data Lineage Visualization')

# Display the legend at the top right
draw_legend()

# Add a chat input at the top
st.header("Ask a Question")
user_input = st.text_input("What is your question?", key="user_question")

if user_input:
    # Process the message
    process_message(user_input)

# Load the data
with st.spinner('Loading data...'):
    df = fetch_lineage_data()

G = build_graph(df)

# Now proceed with the rest of the code, using st.session_state in the sidebar filters

st.sidebar.header('Configuration')

# Add reset button
if st.sidebar.button("Reset Conversation"):
    reset_conversation()

# Check if result_df is in session_state
if 'result_df' in st.session_state and not st.session_state['result_df'].empty:
    result_df = st.session_state['result_df']

    # Determine the route
    route = st.session_state.get('route', 'route1')

    if route == 'route1':
        # All required fields are present
        # Proceed as usual

        # Generate options
        options = []
        for _, row in result_df.iterrows():
            if row['TARGET_TYPE'] == "Tableau Field":
                option_str = (
                    f"{row['TARGET_NAME']} is a {row['TARGET_TYPE']} which is present in "
                    f"{row['SHEET']} of {row['DASHBOARD']} of {row['WORKBOOK']}"
                )
            else:
                option_str = (
                    f"{row['TARGET_NAME']} is a {row['TARGET_TYPE']} which is present in "
                    f"{row['TARGET_TABLE']} of {row['TARGET_DATABASE']}"
                )
            options.append(option_str)

        selected_option = st.selectbox(
            "Select an option:", options, key='selected_option'
        )

        if selected_option:
            selected_record_index = options.index(selected_option)
            selected_record = result_df.iloc[selected_record_index]

            # Extract relevant values from selected_record
            selected_target_type = selected_record['TARGET_TYPE']
            selected_workbook = selected_record.get('WORKBOOK')
            selected_dashboard = selected_record.get('DASHBOARD')
            selected_sheet = selected_record.get('SHEET')
            selected_target_name = selected_record['TARGET_NAME']
            selected_target_table = selected_record.get('TARGET_TABLE')
            selected_target_database = selected_record.get('TARGET_DATABASE')

            # Set st.session_state values
            st.session_state['selected_target_type'] = selected_target_type
            st.session_state['selected_workbook'] = selected_workbook
            st.session_state['selected_dashboard'] = selected_dashboard
            st.session_state['selected_sheet'] = selected_sheet
            if selected_target_type == 'Tableau Field':
                st.session_state['selected_fields'] = [selected_target_name]
            else:
                st.session_state['selected_table'] = selected_target_table
                st.session_state['selected_columns'] = [selected_target_name]
                st.session_state['target_database'] = selected_target_database

            # Clear result_df to hide the selectbox after selection
            del st.session_state['result_df']
            del st.session_state['route']

    elif route == 'route2':
        # Required fields are missing
        # Display the results and a message
        st.info("Your prompt doesn't seem to be related to the Data Lineage, but here's what I could find.")
        st.write(result_df)
        # Clear the result_df and route after displaying
        del st.session_state['result_df']
        del st.session_state['route']

# Get unique target types
target_types = df['TARGET_TYPE'].dropna().unique().tolist()
selected_target_type = st.sidebar.selectbox(
    'Select Target Type',
    target_types,
    index=target_types.index(st.session_state.get('selected_target_type', target_types[0])) if st.session_state.get('selected_target_type', None) in target_types else 0,
    key='selected_target_type'
)

df_filtered = df[df['TARGET_TYPE'] == selected_target_type]

if selected_target_type == 'Tableau Field':
    # Workbooks
    workbooks = df_filtered['WORKBOOK'].dropna().unique().tolist()
    selected_workbook = st.sidebar.selectbox(
        'Select a Workbook',
        workbooks,
        index=workbooks.index(st.session_state.get('selected_workbook', workbooks[0])) if st.session_state.get('selected_workbook', None) in workbooks else 0,
        key='selected_workbook'
    )

    df_filtered_workbook = df_filtered[df_filtered['WORKBOOK'] == selected_workbook]

    # Dashboards
    dashboards = df_filtered_workbook['DASHBOARD'].dropna().unique().tolist()
    selected_dashboard = st.sidebar.selectbox(
        'Select a Dashboard',
        dashboards,
        index=dashboards.index(st.session_state.get('selected_dashboard', dashboards[0])) if st.session_state.get('selected_dashboard', None) in dashboards else 0,
        key='selected_dashboard'
    )

    df_filtered_dashboard = df_filtered_workbook[df_filtered_workbook['DASHBOARD'] == selected_dashboard]

    # Sheets
    sheets = df_filtered_dashboard['SHEET'].dropna().unique().tolist()
    selected_sheet = st.sidebar.selectbox(
        'Select a Sheet',
        sheets,
        index=sheets.index(st.session_state.get('selected_sheet', sheets[0])) if st.session_state.get('selected_sheet', None) in sheets else 0,
        key='selected_sheet'
    )

    df_filtered_sheet = df_filtered_dashboard[df_filtered_dashboard['SHEET'] == selected_sheet]

    # Multi-select for Fields
    target_fields = df_filtered_sheet['TARGET_NAME'].dropna().unique().tolist()
    selected_fields = st.sidebar.multiselect(
        'Select Fields',
        target_fields,
        default=st.session_state.get('selected_fields', target_fields[:1]),
        key='selected_fields'
    )

    # Depth Control
    max_upstream_depth = st.sidebar.number_input('Max Upstream Depth', min_value=1, max_value=10, value=5, step=1)
    max_downstream_depth = st.sidebar.number_input('Max Downstream Depth', min_value=1, max_value=10, value=5, step=1)

    if selected_fields:
        selected_nodes = [
            create_node_id('Tableau Field', field) for field in selected_fields
        ]
        missing_nodes = [node for node in selected_nodes if node not in G.nodes()]

        if missing_nodes:
            st.error(f"Selected fields not found in the graph: {', '.join(missing_nodes)}")
        else:
            for node, field in zip(selected_nodes, selected_fields):
                subgraph = get_lineage_subgraph(G, node, max_upstream_depth, max_downstream_depth)
                dot = plot_graph(subgraph, [node], G)

                if dot:
                    # Create a separate expander for each field's graph and metadata
                    with st.expander(f"Lineage for Field: {field}", expanded=True):
                        st.graphviz_chart(dot, use_container_width=True)

                        # Display metadata inside the expander
                        field_metadata = df[
                            (df['TARGET_NAME'] == field) &
                            (df['SHEET'] == selected_sheet) &
                            (df['DASHBOARD'] == selected_dashboard) &
                            (df['WORKBOOK'] == selected_workbook)
                        ]

                        if not field_metadata.empty:
                            field_metadata['TRANSFORMATION'] = field_metadata['TRANSFORMATION'].fillna('One-to-one mapping')
                            metadata_columns = [
                                'WORKBOOK', 'DASHBOARD', 'SHEET', 'TARGET_NAME',
                                'TRANSFORMATION', 'SOURCE_TYPE', 'SOURCE_NAME',
                                'SOURCE_TABLE', 'SOURCE_DATABASE'
                            ]
                            first_row = field_metadata[metadata_columns].iloc[0]
                            metadata_dict = first_row.to_dict()

                            st.subheader("Field Metadata")
                            render_metadata_table(metadata_dict, metadata_display_names)
                        else:
                            st.write("No metadata available for this field.")
else:
    # Target Tables
    target_tables = df_filtered['TARGET_TABLE'].dropna().unique().tolist()
    selected_table = st.sidebar.selectbox(
        'Select a Target Table',
        target_tables,
        index=target_tables.index(st.session_state.get('selected_table', target_tables[0])) if st.session_state.get('selected_table', None) in target_tables else 0,
        key='selected_table'
    )
    df_filtered_table = df_filtered[df_filtered['TARGET_TABLE'] == selected_table]

    # Multi-select for Columns
    target_columns = df_filtered_table['TARGET_NAME'].dropna().unique().tolist()
    selected_columns = st.sidebar.multiselect(
        'Select Target Columns',
        target_columns,
        default=st.session_state.get('selected_columns', target_columns[:1]),
        key='selected_columns'
    )

    # Get target database
    if 'target_database' in st.session_state:
        target_database = st.session_state['target_database']
    else:
        target_database = df_filtered_table['TARGET_DATABASE'].iloc[0]
        st.session_state['target_database'] = target_database

    # Depth Control
    max_upstream_depth = st.sidebar.number_input('Max Upstream Depth', min_value=1, max_value=10, value=5, step=1)
    max_downstream_depth = st.sidebar.number_input('Max Downstream Depth', min_value=1, max_value=10, value=5, step=1)

    if selected_columns:
        selected_nodes = [
            create_node_id('Database Column', col, target_database, selected_table)
            for col in selected_columns
        ]
        missing_nodes = [node for node in selected_nodes if node not in G.nodes()]

        if missing_nodes:
            st.error(f"Selected columns not found in the graph: {', '.join(missing_nodes)}")
        else:
            for node, column in zip(selected_nodes, selected_columns):
                subgraph = get_lineage_subgraph(G, node, max_upstream_depth, max_downstream_depth)
                dot = plot_graph(subgraph, [node], G)

                if dot:
                    # Create a separate expander for each column's graph and metadata
                    with st.expander(f"Lineage for Column: {column}", expanded=True):
                        st.graphviz_chart(dot, use_container_width=True)

                        # Display metadata inside the expander
                        column_metadata = df[
                            (df['TARGET_NAME'] == column) &
                            (df['TARGET_TABLE'] == selected_table) &
                            (df['TARGET_DATABASE'] == target_database)
                        ]

                        if not column_metadata.empty:
                            column_metadata['TRANSFORMATION'] = column_metadata['TRANSFORMATION'].fillna('One-to-one mapping')
                            metadata_columns = [
                                'TARGET_DATABASE', 'TARGET_TABLE', 'TARGET_NAME',
                                'TRANSFORMATION', 'SOURCE_TYPE', 'SOURCE_NAME',
                                'SOURCE_TABLE', 'SOURCE_DATABASE'
                            ]
                            first_row = column_metadata[metadata_columns].iloc[0]
                            metadata_dict = first_row.to_dict()

                            st.subheader("Column Metadata")
                            render_metadata_table(metadata_dict, metadata_display_names)
                        else:
                            st.write("No metadata available for this column.")
