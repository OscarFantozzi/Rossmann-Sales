import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime as dt

class Rossmann( object ):
    def __init__( self ):
        self.home_path = r''
        self.competition_distance_scaler   = pickle.load( open( self.home_path + r'../parameter/competition_distance_scaler.pkl' , 'rb' ) )
        self.competition_time_month_scaler = pickle.load( open( self.home_path + r'../parameter/competition_time_month_scaler.pkl' , 'rb' ) )
        self.promo_time_week_scaler        = pickle.load( open( self.home_path + r'../parameter/promo_time_week_scaler.pkl', 'rb' ) )
        self.year_scaler                   = pickle.load( open( self.home_path + r'../parameter/year_scaler.pkl' , 'rb' ) )
        self.store_type_scaler             = pickle.load( open( self.home_path + r'../parameter/store_type_scaler.pkl' , 'rb' ) )
        
    def data_cleaning( self, df1 ):
        # crio uma lista com o nome das colunas antigos
        cols_old = ['Store', 'DayOfWeek', 'Date','Open', 'Promo',
                   'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                   'CompetitionDistance', 'CompetitionOpenSinceMonth',
                   'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                   'Promo2SinceYear', 'PromoInterval']

        # coloco as colunas no estilo snake_case
        snake_case = lambda x : inflection.underscore( x ) # defino a lambda function

        # aplico o map para cada nome cols_old e crio a lista cols_new
        cols_new = list( map( snake_case, cols_old ) )

        # renomeio as coluanas do dataset
        df1.columns = cols_new

        # alterando o tipo da coluna date de int para date 64 
        df1['date'] = pd.to_datetime( df1['date'] )

        #competition_distance - Considero que na é igual a não competição. Se eu subsituir na por uma distancia muito grande seria o mesmo que dizer que não existe competição ( o valor máximo df1['competition_distance'] é 75860.0 )
        df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000 if math.isnan( x ) else x )

        #competition_open_since_month - Considero que quando tiver "na" o competidor abriu no mesmo mes da coluna date ( data da venda ) -- para cada linha "na" pego o mes da coluna date e substituo na coluna competition_open_since_month
        df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis = 1 )

        #competition_open_since_year - mesma lógica do competition_open_since_month mas aplicado ao ano
        df1['competition_open_since_year'] = df1.apply( lambda x : x['date'].year if math.isnan( x['competition_open_since_year']                                                 ) else x['competition_open_since_year'], axis = 1)

        #promo2_since_week - Vou considerar que onde tiver "na" a loja não participou da promoção. Então vou substituir o na pela semana da data na coluna date ( data da venda)
        df1['promo2_since_week'] = df1.apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'] , axis = 1)

        #promo2_since_year - mesma lógica da promo2_since_week só que substituindo pelo ano
        df1['promo2_since_year'] = df1.apply( lambda x : x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'] , axis = 1 )

        #promo_interval
        month_map = { 1 : 'Jan', 2 :  'Feb', 3 : 'Mar', 4 : 'Apr', 5 : 'May', 6 : 'Jun', 7 : 'Jul', 8 : 'Aug', 9 : 'Sept', 10 : 'Oct', 11 : 'Nov', 12 : 'Dec'  } # crio um dicionario com os meses de acordo com os valores da coluna promo_interval

        df1['month_map'] = df1['date'].dt.month.map( month_map ) # substituo o numero do mes pelo nome

        df1['promo_interval'].fillna( 0 , inplace = True ) # substituo os na por 0

        df1['is_promo'] = df1[['promo_interval','month_map']].apply(lambda x : 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis = 1) # para cada linha verifico se o valor da coluna month_map existe na coluna promo_interval, se existir retorno 1 ( participa da promoção ) se não existir 0 ( não participa )

        # alterando o tipo da coluna competition_open_since_month  de float64 para int
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( 'int64' )

        # alterando o tipo da coluna competition_open_since_year de float64 para int
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( 'int64' )

        # alterando o tipo da coluna promo2_since_week de float 64para int
        df1['promo2_since_week'] = df1['promo2_since_week'].astype( 'int64' )

        # alterando o tipo da coluna promo2_since_year de float64 para int
        df1['promo2_since_year'] = df1['promo2_since_year'].astype( 'int64' ) 
        
        return df1
    
    def feature_engineering( self, df2 ):
        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week

        # year week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        #=======================================================================================================================#

        # competition since - nova coluna que leva em consideração o ano e o mes da competição
        df2['competition_since'] = df2.apply( lambda x: dt.datetime( year = x['competition_open_since_year'], month = x['competition_open_since_month'], day = 1 ) , axis= 1) 

        # competition time month - Tempo de competição em meses
        df2['competition_time_month'] = ( (df2['date'] - df2['competition_since']) / 30 ).apply( lambda x: x.days ).astype( 'int64' )

        # promo since - Desde quando a promoção ativa
        df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )
        df2['promo_since'] = df2['promo_since'].apply( lambda x : dt.datetime.strptime( x + '-1', '%Y-%W-%w' ) - dt.timedelta(  days = 7 ) )
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since']) / 7).apply( lambda x: x.days ).astype( 'int64' )

        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x : 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        # state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x : 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # filtros as variáveis que não terei no momento da predição
        df2 = df2.loc[ ( df2['open'] != 0 ) , : ]

        # filtragem de colunas - as colunas que não vou passar no momento da predição. Neste caso customers
        cols_drop = ['open', 'promo_interval', 'month_map' ]

        df2 = df2.drop( cols_drop, axis = 1  )
        
        return df2
    
    def data_preparation( self, df5 ):
        # competition distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform( df5[['competition_distance']].values )
        
        # competition time month
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform( df5[['competition_time_month']].values )
        
        # promo time week
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform( df5[['promo_time_week']].values )
        
        # year
        df5['year'] = self.year_scaler.fit_transform( df5[['year']].values )
        
        # state holiday - One Hot Encoding
        df5 = pd.get_dummies( df5 , prefix = 'state_holiday', columns = ['state_holiday'] )

        # store_type - Labelencoder
        df5['store_type'] = self.store_type_scaler.fit_transform( df5['store_type'] )

        # assorment - ordinal encoding
        assortment_dict = { 'basic' : 1, 'extra' : 2, 'extended' : 3 }
        df5['assortment'] = df5['assortment'].map( assortment_dict )
        
        # day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * (2. * np.pi / 7 ) ) )
        df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * (2. * np.pi / 7 ) ) )

        # month
        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi / 12 ) ) )
        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi / 12 ) ) )

        # day
        df5['day_sin'] = df5['day'].apply( lambda x : np.sin( x * ( 2. * np.pi / 30 ) ) )
        df5['day_cos'] = df5['day'].apply( lambda x : np.cos( x * (2. * np.pi / 30) ) )

        # week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi / 52 )  ) )
        df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * (2. * np.pi / 52 ) ) )
        
        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance','competition_open_since_month','competition_open_since_year','promo2','promo2_since_week','promo2_since_year','competition_time_month','promo_time_week','day_of_week_sin',
                        'day_of_week_cos','month_cos','month_sin','day_sin','day_cos','week_of_year_cos','week_of_year_sin']
        
        return df5[ cols_selected ]  
    
    def get_prediction( self, model, original_data, test_data ):
        # faz a predicao
        pred = model.predict( test_data )
        
        # anexa ao df original
        original_data['predictions'] = np.expm1( pred )
        
        return original_data.to_json( orient = 'records', date_format = 'iso' )