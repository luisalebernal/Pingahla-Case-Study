import dash
from dash import html
from dash import dcc
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from datetime import datetime


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
app.css.append_css({'external_url': '/static/reset.css'})
app.server.static_folder = 'static'
server = app.server

app.layout = dbc.Container([
    dcc.Store(id='store-data-part-costs', storage_type='memory'),  # 'local' or 'session'
    dcc.Store(id='store-data-part-numbers', storage_type='memory'),  # 'local' or 'session'
    dcc.Store(id='store-data-add-info', storage_type='memory'),  # 'local' or 'session'

    dcc.Interval(
        id='my_interval',
        disabled=False,
        interval=1 * 1000,
        n_intervals=0,
        max_intervals=1
    ),
    dbc.Row([
        dbc.Col(html.H5(
            '"Any sufficiently advanced technology is indistinguishable from magic." - Arthur C. Clarke '),
                style={'color': "green", 'font-family': "Franklin Gothic"}, width=7),

    ]),
    dbc.Row([
        dbc.Col(html.H1(
            "Case Study - Data Engineer - Pingahla",
            style={'textAlign': 'center', 'color': '#082255', 'font-family': "Franklin Gothic"}), width=12, )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem([
                    html.H5([
                                'The next Dashboard presents the main results of the Luxury Cars Case Study for the Data Engineer role for Pingahla.'])

                ], title="Introduction"),
            ], start_collapsed=True, style={'textAlign': 'left', 'color': '#082255', 'font-family': "Franklin Gothic"}),

        ], style={'color': '#082255', 'font-family': "Franklin Gothic"}),
    ]),

    dbc.Row([
        dbc.Tabs([
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Row(html.H2(['Summary per Car']),
                                                style={'color': '#082255', 'font-family': "Franklin Gothic"})
                                    ])
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Car:",
                                            id="car-dropdown-target",
                                            color="info",
                                            style={'font-family': "Franklin Gothic"},
                                            className="me-1",
                                            n_clicks=0,
                                        ),
                                        dbc.Popover(
                                            "Is the car entered by the user to get its summary.",
                                            target="car-dropdown-target",
                                            body=True,
                                            trigger="hover",
                                            style={'font-family': "Franklin Gothic"}
                                        ),
                                    ], width=2, align='center', className="d-grid gap-2"),

                                    dbc.Col(
                                        dbc.Spinner(children=[dcc.Dropdown(id='car-dropdown', style={'font-family': "Franklin Gothic"})], size="lg",
                                                    color="primary", type="border", fullscreen=True, ),
                                                                                xs=3, sm=3, md=3, lg=2, xl=2, align='center'),

                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Number of Parts:",
                                            id="no-parts-car-target",
                                            color="primary",
                                            style={'font-family': "Franklin Gothic"},
                                            className="me-1",
                                            n_clicks=0,
                                        ),
                                        dbc.Popover(
                                            "It is the number of parts of the given car.",
                                            target="no-parts-car-target",
                                            body=True,
                                            trigger="hover",
                                            style={'font-family': "Franklin Gothic"}
                                        ),
                                    ], width=2, align='center', className="d-grid gap-2"),
                                    dbc.Col([
                                        html.Div(id='no-parts-car', style={'font-family': "Franklin Gothic"})
                                    ], xs=2, sm=2, md=2, lg=2, xl=2, style={'textAlign': 'center'}, align='center'),
                                    dbc.Col([
                                        dbc.Button(
                                            "Total Cost [USD]:",
                                            id="total-cost-car-target",
                                            color="success",
                                            style={'font-family': "Franklin Gothic"},
                                            className="me-1",
                                            n_clicks=0,
                                        ),
                                        dbc.Popover(
                                            "It is the total cost for producing the car (sum of the costs from all its parts).",
                                            target="total-cost-car-target",
                                            body=True,
                                            trigger="hover",
                                            style={'font-family': "Franklin Gothic"}
                                        ),
                                    ], width=2, align='center', className="d-grid gap-2"),
                                    dbc.Col([
                                        html.Div(id='total-cost-car', style={'font-family': "Franklin Gothic"})
                                    ], xs=2, sm=2, md=2, lg=2, xl=2, style={'textAlign': 'center'}, align='center'),
                                    dbc.Col([
                                        dbc.Button(
                                            "Avg. Part Price [USD]:",
                                            id="avg-price-car-target",
                                            color="primary",
                                            style={'font-family': "Franklin Gothic"},
                                            className="me-1",
                                            n_clicks=0,
                                        ),
                                        dbc.Popover(
                                            "It is the average price of the parts for the production of the car.",
                                            target="avg-price-car-target",
                                            body=True,
                                            trigger="hover",
                                            style={'font-family': "Franklin Gothic"}
                                        ),
                                    ], width=2, align='center', className="d-grid gap-2"),
                                    dbc.Col([
                                        html.Div(id='avg-price-car', style={'font-family': "Franklin Gothic"})
                                    ], xs=2, sm=2, md=2, lg=2, xl=2, style={'textAlign': 'center'}, align='center'),

                                ]),


                                dbc.Row(dbc.Spinner(dcc.Graph(id='fig-world-parts'), size="lg",
                                                    color="primary", type="border", fullscreen=True, )),
                                dbc.Row(dbc.Spinner(dcc.Graph(id='fig-world-cost'), size="lg",
                                                    color="primary", type="border", fullscreen=True, )),

                            ])
                        ]),

                    ]),

                ]),
            ], label="Summary per Car", label_style={'color': '#082255', 'font-family': "Franklin Gothic"}),
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Row(html.H2(['General Summary']),
                                                style={'color': '#082255', 'font-family': "Franklin Gothic"})
                                    ])
                                ]),

                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Avg. Number of Parts:",
                                            id="no-parts-general-target",
                                            color="primary",
                                            style={'font-family': "Franklin Gothic"},
                                            className="me-1",
                                            n_clicks=0,
                                        ),
                                        dbc.Popover(
                                            "It is the average number of parts from all the cars.",
                                            target="no-parts-general-target",
                                            body=True,
                                            trigger="hover",
                                            style={'font-family': "Franklin Gothic"}
                                        ),
                                    ], width=2, align='center', className="d-grid gap-2"),
                                    dbc.Col([
                                        html.Div(id='no-parts-general', style={'font-family': "Franklin Gothic"})
                                    ], xs=2, sm=2, md=2, lg=2, xl=2, style={'textAlign': 'center'}, align='center'),
                                    dbc.Col([
                                        dbc.Button(
                                            "Avg. Total Cost [USD]:",
                                            id="total-cost-general-target",
                                            color="success",
                                            style={'font-family': "Franklin Gothic"},
                                            className="me-1",
                                            n_clicks=0,
                                        ),
                                        dbc.Popover(
                                            "It is the average total cost for producing from all the cars.",
                                            target="total-cost-general-target",
                                            body=True,
                                            trigger="hover",
                                            style={'font-family': "Franklin Gothic"}
                                        ),
                                    ], width=2, align='center', className="d-grid gap-2"),
                                    dbc.Col([
                                        html.Div(id='total-cost-general', style={'font-family': "Franklin Gothic"})
                                    ], xs=2, sm=2, md=2, lg=2, xl=2, style={'textAlign': 'center'}, align='center'),
                                    dbc.Col([
                                        dbc.Button(
                                            "Avg. Part Price [USD]:",
                                            id="avg-price-general-target",
                                            color="primary",
                                            style={'font-family': "Franklin Gothic"},
                                            className="me-1",
                                            n_clicks=0,
                                        ),
                                        dbc.Popover(
                                            "It is the average price of the parts for the production from all the cars.",
                                            target="avg-price-general-target",
                                            body=True,
                                            trigger="hover",
                                            style={'font-family': "Franklin Gothic"}
                                        ),
                                    ], width=2, align='center', className="d-grid gap-2"),
                                    dbc.Col([
                                        html.Div(id='avg-price-general', style={'font-family': "Franklin Gothic"})
                                    ], xs=2, sm=2, md=2, lg=2, xl=2, style={'textAlign': 'center'}, align='center'),

                                ]),
                                dbc.Row(dbc.Spinner(dcc.Graph(id='fig-count'), size="lg",
                                                    color="primary", type="border", fullscreen=True, )),
                                dbc.Row(dbc.Spinner(dcc.Graph(id='fig-total-cost'), size="lg",
                                                    color="primary", type="border", fullscreen=True, )),
                                dbc.Row(dbc.Spinner(dcc.Graph(id='fig-avg-price'), size="lg",
                                                    color="primary", type="border", fullscreen=True, )),


                            ])
                        ]),

                    ]),

                ]),
            ], label="General Summary", label_style={'color': '#082255', 'font-family': "Franklin Gothic"}),

        ]),
    ]),

])


@app.callback(
    Output(component_id='store-data-part-costs', component_property='data'),
    Output(component_id='store-data-part-numbers', component_property='data'),
    Output(component_id='store-data-add-info', component_property='data'),
    Output(component_id='car-dropdown', component_property='options'),

    Output(component_id='car-dropdown', component_property='value'),


    Input('my_interval', 'n_intervals'),
)
def dropdownTiempoReal(value_intervals):

    # Import csv's
    df_part_costs = pd.read_csv('Part_Costs.csv', lineterminator='\r')
    df_part_numbers = pd.read_csv('Part_Numbers.csv', lineterminator='\r')
    df_add_info = pd.read_csv('Additional_Info.csv', lineterminator='\r')


    # Sort by car number and identify unique cars
    df_part_numbers = df_part_numbers.sort_values(by=['car'])
    print(df_part_numbers)
    types_of_cars = df_part_numbers['car']
    types_of_cars = types_of_cars.drop_duplicates()
    print(types_of_cars)

    # OBS: the car number can be in numerical form or in sting.

    # The car number is transformed only to its numeric value
    dict_map_numbers = {'CAR 1': '1', 'CAR ONE': '1', 'ONE': '1', 'Car 1': '1', 'Car 2': '2', 'TWO': '2', 'TWO ': '2',
                        'Car 3': '3', 'THREE': '3', '(THREE)': '3', 'Car 4': '4', 'Car 4 ': '4', 'Four': '4',
                        'CAR 5': '5', 'FIVE': '5', 'FIVE ': '5', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5',}
    df_part_numbers['car'] = df_part_numbers['car'].map(dict_map_numbers)
    print(df_part_numbers)
    types_of_cars = df_part_numbers['car']
    types_of_cars = types_of_cars.drop_duplicates()
    print(types_of_cars)

    # Dropdown options and value are generated
    carDD = df_part_numbers["car"]
    carDD = carDD.drop_duplicates()
    carDD = carDD.sort_values()
    carDD = carDD.dropna()

    carDD1 = carDD[0]

    return df_part_costs.to_dict('records'), df_part_numbers.to_dict('records'), df_add_info.to_dict('records'),\
           carDD, carDD1,\


@app.callback(
    Output(component_id='no-parts-car', component_property='children'),
    Output(component_id='total-cost-car', component_property='children'),
    Output(component_id='avg-price-car', component_property='children'),
    Output(component_id='fig-count', component_property="figure"),
    Output(component_id='fig-total-cost', component_property="figure"),
    Output(component_id='fig-avg-price', component_property="figure"),
    Output(component_id='fig-world-parts', component_property="figure"),
    Output(component_id='fig-world-cost', component_property="figure"),
    Output(component_id='no-parts-general', component_property='children'),
    Output(component_id='total-cost-general', component_property='children'),
    Output(component_id='avg-price-general', component_property='children'),


    Input(component_id='store-data-part-costs', component_property='data'),
    Input(component_id='store-data-part-numbers', component_property='data'),
    Input(component_id='store-data-add-info', component_property='data'),
    Input(component_id='car-dropdown', component_property='value'),
)
def dashboard(data1, data2, data3, value_car):

    ######### Car Summary #########

    # Dataframes are imported
    df_part_costs = pd.DataFrame(data1)
    df_part_numbers = pd.DataFrame(data2)
    df_add_info = pd.DataFrame(data3)

    # Counts hoy many PART_NO from part_cost table are NaN values
    no_nan_PART_NO = df_part_costs['PART_NO']
    no_nan_PART_NO = no_nan_PART_NO.isna().sum()
    print('no_nan_PART_NO')
    print(no_nan_PART_NO)

    # Part numbers and part costs are joined in order to know the parts of each car
    df_merged = pd.merge(df_part_costs, df_part_numbers, on=["PART_NO"])
    df_merged = df_merged.loc[:, ["car", "PART_NO", "COST", "CURRENCY", 'PURCH_ORDER', 'RAISED_DATE']]
    df_merged = df_merged.sort_values(by=['car'])

    # It is searched how many parts does not have a currency or cost
    no_nan_currency = df_merged['CURRENCY']
    no_nan_currency = no_nan_currency.isna().sum()
    no_nan_cost = df_merged['COST']
    no_nan_cost = no_nan_cost.isna().sum()

    print('AQUI df_merged')
    print(df_merged)
    print(df_part_costs)
    print(no_nan_currency)
    print(no_nan_cost)

    # OBS: all parts have a cost but 12114 from 19738 total merged rows does not have a currency! The currency shall be found with
    # the Additional info table

    # The column PURCH_ORDER of the additional info table starts with "\n", which shall be removed
    df_add_info = df_add_info.apply(lambda x: x.str.replace('\n', ''))

    # The part numbers and part costs joined table is joined with the additional info table to know the currency of each part
    # based on the supplier origin
    df_merged2 = pd.merge(df_merged, df_add_info, how="left", on=["PURCH_ORDER"])
    df_merged2 = df_merged2.loc[:,
                 ["car", "PART_NO", "COST", "CURRENCY_x", 'CURRENCY_y', 'SUPPLIER_ORIGIN', 'RAISED_DATE']]

    no_nan_currencyx = df_merged2['CURRENCY_x']
    no_nan_currencyx = no_nan_currencyx.isna().sum()
    no_nan_currencyy = df_merged2['CURRENCY_y']
    no_nan_currencyy = no_nan_currencyy.isna().sum()

    # Only the non NaN values from columns CURRENCY_x and CURRENCY_y are taken
    df_merged2['CURRENCY'] = df_merged2.filter(like='CURRENCY').max(1)
    df_merged2 = df_merged2.loc[:, ["car", "PART_NO", "COST", "CURRENCY", 'SUPPLIER_ORIGIN', 'RAISED_DATE']]

    print(df_merged)
    print(df_add_info)
    print('AQUI df_merged2')
    print(df_merged2)
    print(no_nan_currencyx)
    print(no_nan_currencyy)

    # OBS: after joining tables there still are 12327 NaN values from 20119 rows of currency. The currencies will be obtained
    # based on the supplier origin

    # Get how many different supplier origins and currencys are there
    types_of_SO = df_merged2['SUPPLIER_ORIGIN']
    types_of_SO = types_of_SO.drop_duplicates()
    types_of_currency = df_merged2['CURRENCY']
    types_of_currency = types_of_currency.drop_duplicates()
    print(types_of_SO)
    print(types_of_currency)

    # Map the currency based on the supplier origin
    dict_map = {'UK': 'GBP', 'Germany': 'EUR', 'France': 'EUR', 'USA': 'USD', 'Sweden': 'SEK', 'Japan': 'JPY',
                'CZK': 'CZK',
                'Australia': 'AUD'}
    df_merged2['CURRENCY_BSU'] = df_merged2['SUPPLIER_ORIGIN'].map(dict_map)
    print(df_merged2)

    # Only the non NaN values from columns CURRENCY and CURRENCY_BSU are taken
    df_merged2['CURRENCY'].replace(to_replace=[None], value=np.nan, inplace=True)
    df_merged2['CURRENCY_FINAL'] = df_merged2['CURRENCY'].combine_first(df_merged2['CURRENCY_BSU'])

    no_nan_currencyF = df_merged2['CURRENCY_FINAL']
    no_nan_currencyF = no_nan_currencyF.isna().sum()

    df_final = df_merged2.loc[:, ["car", "PART_NO", "COST", "CURRENCY_FINAL", 'SUPPLIER_ORIGIN', 'RAISED_DATE']]

    print(df_merged2)
    print('no_nan_currencyF')
    print(no_nan_currencyF)
    print(df_final)

    # OBS: only 32 rows from 20119 of currency are NaN values! Huge improvement!

    # Currencies will be transformed to USD only
    # Creates column with currency in USD
    dict_map_currency = {'GBP': 1.20, 'EUR': 1.07, 'USD': 1, 'SEK': 0.096, 'JPY': 0.0075, 'CZK': 0.045, 'AUD': 0.69,
                         'ZAR': 0.055}
    df_final['COST'] = df_final['COST'].astype(float)
    df_final['RAISED_DATE'] = df_final['RAISED_DATE'].apply(lambda z: datetime.strptime(z, "%m/%d/%y"))
    df_final['RATE_USD'] = df_final['CURRENCY_FINAL'].map(dict_map_currency)
    df_final['COST_USD'] = df_final['COST'] * df_final['RATE_USD']

    df_final = df_final.loc[:, ["car", "COST", "CURRENCY_FINAL", 'SUPPLIER_ORIGIN', 'RATE_USD', 'COST_USD']]
    # df_final = df_final.loc[:,["car", "PART_NO", "COST", "CURRENCY_FINAL", 'SUPPLIER_ORIGIN', 'RATE_USD', 'COST_USD']]

    print('df_final aca2')
    print(df_final)

    # OBS: now the statistical analysis can be made

    # Calculates total number of parts
    df_final_car = df_final[df_final['car'] == value_car]
    noPartsCar = df_final_car.shape[0]

    # Calculates total cost of parts in USD
    totalCostCar = df_final_car['COST_USD'].sum()
    totalCostCar = round(totalCostCar, 0)

    # Calculates average price of parts in USD
    avgPriceCar = df_final_car['COST_USD'].mean()
    avgPriceCar = round(avgPriceCar, 0)

    ######### General Summary #########

    # Groups by car and calculates number of cars, total cost of car and avg price of parts.
    # Calculates average number of parts, total cost and average part price from all cars.

    df_final_noParts_graph = df_final.groupby("car")["car"].count()
    x_noParts = df_final_noParts_graph.values
    y_noParts = df_final_noParts_graph.index
    avgNoPartsGen = round(x_noParts.mean(), 0)

    df_final_totalCost_graph = df_final.groupby("car")["COST_USD"].sum()
    x_totalCost = df_final_totalCost_graph.values
    y_totalCost = df_final_totalCost_graph.index
    avgTotCostGen = round(x_totalCost.mean(), 0)


    df_final_avgPrice_graph = df_final.groupby("car")["COST_USD"].mean()
    x_avgPrice = df_final_avgPrice_graph.values
    y_avgPrice = df_final_avgPrice_graph.index
    avgAvgPriceGen = round(x_avgPrice.mean(), 0)


    print('df_final_noParts_graph')
    print(df_final_noParts_graph)
    print(df_final_totalCost_graph)
    print(df_final_avgPrice_graph)

    # Defines function to create bar charts
    def create_bar_chart(x, y, title, x_title, y_title):
        figure = px.bar(x=x, y=y)

        figure.update_layout(
            font_family="Franklin Gothic",
            title_font_family="Franklin Gothic",
            barmode='group',
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,

        )

        return figure

    # Uses function to create the bar charts
    fig_bar_noParts = create_bar_chart(x_noParts, y_noParts, "Number of Parts per Car", "Count", "Car")
    fig_bar_totalCost = create_bar_chart(x_totalCost, y_totalCost, "Total Cost per Car", "Total Cost [USD]", "Car")
    fig_bar_avgPrice = create_bar_chart(x_avgPrice, y_avgPrice, "Average Price of Parts per Car", "Avg. Price [USD]", "Car")


    ##### World Maps #####

    # Maps the name of the country
    dict_map_country = {'CZK': 'Czech Republic', 'UK': 'United Kingdom', 'USA': 'United States', 'Australia':'Australia'
                        ,'France':'France', 'Germany':'Germany', 'Japan':'Japan', 'Sweden':'Sweden'}
    df_final['SUPPLIER_ORIGIN'] = df_final['SUPPLIER_ORIGIN'].map(dict_map_country)

    # Maps the continent of the country
    dict_map_continent = {'Czech Republic': 'Europe', 'United Kingdom': 'Europe', 'United States': 'North America',
                          'Australia':'Oceania','France':'Europe', 'Germany':'Europe', 'Japan':'Oceania',
                          'Sweden':'Europe'}
    df_final['Continent'] = df_final['SUPPLIER_ORIGIN'].map(dict_map_continent)

    # Maps the iso alpha 3 code of the country
    dict_map_isoAlpha = {'Czech Republic': 'CZE', 'United Kingdom': 'GBR', 'United States': 'USA',
                          'Australia':'AUS','France':'FRA', 'Germany':'DEU', 'Japan':'JPN',
                          'Sweden':'Europe'}
    df_final['iso_alpha'] = df_final['SUPPLIER_ORIGIN'].map(dict_map_isoAlpha)

    # Creates dataframe to plot the world map
    df_final_car = df_final[df_final['car'] == value_car]
    df_final_world_map_group = df_final_car.groupby("SUPPLIER_ORIGIN")["car"].count()
    df_final_world_map = pd.DataFrame()
    df_final_world_map['SUPPLIER_ORIGIN'] = df_final_world_map_group.index
    df_final_world_map['Parts'] = df_final_world_map_group.values
    df_final_world_map['iso_alpha'] = df_final_world_map['SUPPLIER_ORIGIN'].map(dict_map_isoAlpha)

    df_final_world_map_group = df_final_car.groupby("SUPPLIER_ORIGIN")["COST"].sum()
    df_final_world_map['Cost'] = df_final_world_map_group.values


    print('df_final_world_map')
    print(df_final_world_map)

    # Creates world maps
    fig_world_parts = px.choropleth(df_final_world_map, locations="iso_alpha", color="Parts",
                         hover_name="SUPPLIER_ORIGIN",
                         projection="natural earth",
                              title="Number of Parts Distribution",

                                    )

    fig_world_cost = px.choropleth(df_final_world_map, locations="iso_alpha", color="Cost",
                         hover_name="SUPPLIER_ORIGIN",
                         projection="natural earth",
                              title="Cost of Parts Distribution")

    return noPartsCar, totalCostCar, avgPriceCar, fig_bar_noParts, fig_bar_totalCost, fig_bar_avgPrice, fig_world_parts\
            , fig_world_cost, avgNoPartsGen, avgTotCostGen, avgAvgPriceGen





if __name__ == '__main__':
    app.run_server()
