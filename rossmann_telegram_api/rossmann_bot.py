import pandas as pd
import json
import requests
from flask import Flask, request, Response

# # constants
token = '5467220512:AAFXDcO--pup9FlYI_3baZSHNx6W4gjGVaQ'

# # info about bot
# https://api.telegram.org/bot5467220512:AAFXDcO--pup9FlYI_3baZSHNx6W4gjGVaQ/getMe

# # get updates
# https://api.telegram.org/bot5467220512:AAFXDcO--pup9FlYI_3baZSHNx6W4gjGVaQ/getUpdates

# # send messages
# https://api.telegram.org/bot5467220512:AAFXDcO--pup9FlYI_3baZSHNx6W4gjGVaQ/sendMessage?chat_id=5369527365&text=Hi Oscar, I'm doing great

# webhhok updates
# https://api.telegram.org/bot5467220512:AAFXDcO--pup9FlYI_3baZSHNx6W4gjGVaQ/setWebhook?url= https://3183f0b5e069c4.lhr.life

def send_message( chat_id, text ):
    url = 'https://api.telegram.org/bot{}/'.format( token )
    
    url = url + 'sendMessage?chat_id={}'.format( chat_id )
    
    r = requests.post( url, json = {'text' : text } )
    
    print( 'Status Code {}'.format(r.status_code) )
    
    return None
        
def load_data( store_id ):
    # loading test dataset
    df10 = pd.read_csv( r'data\test.csv' )
    df_store_raw = pd.read_csv( r'data\store.csv' )

    # merge
    df_test = pd.merge( df10 , df_store_raw, how = 'left', on = 'Store' )

    # choose store for prediction
    df_test = df_test.loc[df_test['Store'] == store_id, :]
    
    if not df_test.empty:
        # remove closed days
        df_test = df_test.loc[ df_test['Open'] != 0, : ] # remove closed days
        df_test = df_test.loc[~ df_test['Open'].isnull(), : ] # remove nulls on columns open

        # remove columns ID
        df_test = df_test.drop( 'Id', axis = 1 )

        # convert dataframe into json
        data = json.dumps( df_test.to_dict( orient = 'records' ) )
    
    else:
        data = 'error'
    
    return data

def predict( data ):
    # API call
    url = 'https://api-teste-rs-2023.ew.r.appspot.com/rossmann/predict'
    header = {'Content-type' : 'application/json'}

    data = data

    r = requests.post( url, data = data, headers = header ) # response in json
    print( 'Status Code {}'.format( r.status_code ) )

    d1 = pd.DataFrame( r.json(), columns = r.json()[0].keys() )
    
    return d1

def parse_message( message ):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']
    
    store_id = store_id.replace('/', '')
    
    try:
        store_id = int(store_id)
    
    except ValueError:
        
        store_id = 'error'
    
    return chat_id, store_id

# API initialize
app = Flask( __name__ )

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        message = request.get_json()
        
        chat_id, store_id = parse_message( message )
        
        if store_id != 'error':
            # loading data
            data = load_data( store_id ) 
            
            if data != 'error':
                # prediction
                d1 = predict( data )
                # calculation
                d2 = d1[['store','predictions']].groupby('store').sum().reset_index()
                
                # send message
                msg = 'Store Number {} will sell ${:,.2f} in the next 6 weeks'.format(
                    d2.loc[:,'store'].values[0],
                    d2.loc[:,'predictions'].values[0] ) 
                
                send_message(chat_id, msg)
                
                return Response('Ok', status = 200)
                
            else:
                send_message(chat_id, 'Store Not Available')
                return Response('Ok', status = 200 )
        else:
            send_message(chat_id, 'Store ID is wrong')
            return Response( 'Ok', status = 200 )

    else:
        return '<h1>Rossmann Telegram Bot<h1>'

if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run(host = '0.0.0.0', port = port)





