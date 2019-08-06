"""
A Dash GUI.

html.Div(id='intermediate-value', style={'display': 'none'})

TO DO: look up caches for Dash
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import base64
import os

from gui_state import GUIState

# Style.
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
tabs_styles = {
	'height': '44px'
}
tab_style = {
	'borderBottom': '1px solid #d6d6d6',
	'padding': '6px',
	'fontWeight': 'bold'
}
tab_selected_style = {
	'borderTop': '1px solid #d6d6d6',
	'borderBottom': '1px solid #d6d6d6',
	'backgroundColor': '#119DFF',
	'color': 'white',
	'padding': '6px'
}

# App.
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# State.
state = GUIState()

# Make components.
def make_upload_component(n=0):
	return dcc.Upload(id='upload-directory'+str(n),
		children=html.Div([
			'Drag and Drop or ',
			html.A('Select Files')
		]),
		style={'width': '80%','height': '60px','lineHeight': '60px',
			'borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px',
			'textAlign': 'center','margin': '10px'
		},
		multiple=False
	)


# Header
header = html.H1(
	children='VAE GUI',
	style={'textAlign': 'center'}
)

# Image
image = html.Div(
	children=[html.Img(
		src='data:image/png;base64,{}'.format(state.get_background()),
		style={'width': '500px'})
	],
	style={'textAlign': 'center'}
)

# Tabs
h_tabs = html.Div([
	dcc.Tabs(
		id="tabs",
		value='tab-1',
		children=[
			dcc.Tab(label='Project Directories', value='tab-1', style=tab_style,
				selected_style=tab_selected_style),
			dcc.Tab(label='Segmenting', value='tab-2', style=tab_style,
				selected_style=tab_selected_style),
			dcc.Tab(label='Preprocessing', value='tab-3', style=tab_style,
				selected_style=tab_selected_style),
			dcc.Tab(label='Model Training', value='tab-4', style=tab_style,
				selected_style=tab_selected_style),
			dcc.Tab(label='Plotting & Analysis', value='tab-5', style=tab_style,
				selected_style=tab_selected_style),
			dcc.Tab(label='Info', value='tab-6', style=tab_style,
				selected_style=tab_selected_style),
		],
		style=tabs_styles
	),
	html.Div(id='tabs-content')
])

# Many different things under the tabs.

# Footer
footer_text = dcc.Markdown(
	children="""
	___
	```
	# NOTE: TO DO
	if __main__=='main':
		pass


		###
		```

		> The End

	___
	"""
)


# TAB CONTENTS

# Directories tab
directories_tab = html.Div(
	children=[
		dcc.Markdown("## 1) Enter project directories:"),
		html.Div("Audio directory:"),
		dcc.Input(placeholder='Enter a directory...',
			type='text',
			value='',
		),
		html.Div("Segmenting directory:"),
		dcc.Input(placeholder='Enter a directory...',
			type='text',
			value='',
		),
		html.Div("Model directory:"),
		dcc.Input(placeholder='Enter a directory...',
			type='text',
			value='',
		),
		html.Div("Projection directory:"),
		dcc.Input(placeholder='Enter a directory...',
			type='text',
			value='',
		),
		html.Div("Plotting directory:"),
		dcc.Input(placeholder='Enter a directory...',
			type='text',
			value='',
		),
		dcc.Markdown(
		"""
		### or
		## upload a previously saved set of directories:
		"""),
		make_upload_component(1),
		dcc.Markdown(
		"""## 2) Save this collection of directories somewhere."""
		)
	]
)


segmenting_tab = html.Div(
	children=[
		dcc.Markdown("## 1) Tune segmenting parameters:")
	]
)


preprocessing_tab = html.Div(
	children=[
		dcc.Markdown("## 1) Tune preprocessing parameters:"),
		dcc.Checklist(
		id='preprocess_binary_checklist',
		options=[
			{'label': 'Sliding Window', 'value': 'sliding_window'},
			{'label': 'Mel Frequency Spacing', 'value': 'mel'},
			{'label': 'Time Stretch', 'value': 'time_stretch'},
			{'label': 'Within-syllable Normalize', 'value': 'within_syll_normalize'},
		],
		value=[key for key in state.p['binary_preprocess_params'] if state.p[key]]
		),
		dcc.Markdown(
		"""
		### or
		## upload previously saved parameters:
		"""
		),
		make_upload_component(2),
	]
)

training_tab = html.Div(
	children=[
		dcc.Markdown(
		"""
		Select saved model:
		"""
		),
		dcc.Dropdown(
		options=[
			{'label': 'None', 'value': 'None'},
			{'label': 'Montréal', 'value': 'MTL'},
			{'label': 'San Francisco', 'value': 'SF'}
		],
		value=0,
		) ,
	]
)


plotting_tab = html.Div(
	children=[
		dcc.Markdown(
		"""Coming soon!"""
		)
	]
)

info_tab = html.Div(
	children=[
		dcc.Markdown(
		"""
		TO DO
		"""
		)
	]
)


####################
#      LAYOUT      #
####################

app.layout = html.Div(children=[
	header,
	image,
	h_tabs,
	footer_text,
])


####################
#     CALLBACKS    #
####################

@app.callback(Output('tabs-content', 'children'),
	  [Input('tabs', 'value')])
def render_content(tab):
	if tab == 'tab-1':
		return directories_tab
	elif tab == 'tab-2':
		return segmenting_tab
	elif tab == 'tab-3':
		return preprocessing_tab
	elif tab == 'tab-4':
		return training_tab
	elif tab == 'tab-5':
		return plotting_tab
	else:
		return info_tab


# @app.callback(Output('preprocess_binary_checklist', 'value'),
# 	  [Input('preprocess_binary_checklist', 'value')])
# def render_content(tab):



if __name__ == '__main__':
	app.run_server(debug=True)



###
