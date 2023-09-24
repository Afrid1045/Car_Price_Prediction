import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib


# Load the KNN model
RFR = joblib.load('RFR_model.joblib')

st.set_page_config(page_title='Estimated Price of the Car')

num_attribs = ['Levy','prod_year', 'Engine_volume', 'Mileage',  'airbags']
cat_attribs = ['Manufacturer','Model', 'Category', 'Leather_interior', 'Fuel_type', 'Gear_box', 'color']

'''num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder())
])

# Define full preprocessing pipeline using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])
'''

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())])

full_pipeline = ColumnTransformer([
    ('num',num_pipeline, num_attribs),
    ('cat',OneHotEncoder(), cat_attribs)  
])




def main():
    # Set page title and favicon
    # Set app title
    st.title('Estimated Price of the Car')
    
    # Create form to take user input
    with st.form(key='car_price_prediction_form'):
        # Add form fields for user input
        st.write('Please fill in the following details to check estimated price of the car:')
        Model = st.selectbox('Select your car:', options=['RX 450',	'Equinox',	'FIT',	'Escape',	'Santa FE',	'Prius',	'Sonata',	'RX 350',	'E 350',	'Transit',	'Vectra',	'CHR',	'Elantra',	'Camry',	'E 220',	'Vito',	'Cayenne',	'X5',	'Grand Cherokee',	'H1',	'Jetta',	'Tacoma',	'Prius C',	'Aqua',	'Escape Hybrid',
                                                        	'Civic',	'Q7',	'Megane 1.5CDI',	'E 300',	'Q5',	'C 180',	
                                                            'Juke',	'535',	'Cruze LT',	'Fusion',	'VOXY',	'A 160',	'Tucson',	
                                                            'Vitz',	'Captiva',	'Mustang',	'ML 350',	'Highlander',	'Yaris',	
                                                            'Cr-v',	'Cruze',	'Orlando',	'520 Vanos',	'Forester',	'Lacetti',	
                                                            '428 Sport Line',	'Patrol',	'E 320',	'Genesis',	'911',	'Sprinter',	
                                                            'Focus SE',	'Picanto',	'328',	'Airtrek',	'Lancer',	'Korando',	'Clio',	
                                                            '616',	'C 220',	'Serena',	'RAV 4',	'Pajero',	'Volt',	'TERRAIN',	
                                                            'Hr-v EX',	'500',	'Legacy',	'Elantra sport limited',	'Sienna',	'A 170',	'REXTON',	'Carnival grand',	'QX60',	'Passat',	'1000',	'50',	'C 250',	'Vitz funkargo',	'A6',	'Pathfinder',	'Delica',	'Golf',	'Vaneo',	'Patrol Y60',	'A7',	'Explorer',	'I30',	'Altima',	'Grand Vitara',	'CT 200h',	
                                                            'Veloster',	'RAV 4 XLE Sport',	'Sienta',	'Avalon LIMITED',	'Cerato K3',	'CX-7',	'Astra G',	'Ist',	'Corolla',	'Rogue',	'MPV',	'GLA 250',	'530',	'Sharan',	'Tiida',	'C 300',	'Actyon',	'GX 460',	'Elgrand',	'C 350',	'CLS 500',	'S 350',	'RAV 4 Le',	'Zafira',	'Vectra b',	'C 200',	'Astra',	'323',	'E 350 áƒáƒ›áƒ’',	'CLK 320',	'Avalon',	'ML 250',	'330',	
                                                            'Colt Lancer',	'318',	'Outlander',	'Camry SE',	'E 200',	'GLE 350',	'Malibu',	'TL',	'Insight',	'Stream',	'I',	'GTI',	'Colt',	'Pajero Mini',	'750',	'RAV 4 s p o r t',	'Outback',	'GX 470',	'Fusion Titanium',	'Jimny',	'Aveo',	'X6',	'Aqua S',	'1111',	'Laguna',	'Optima',	'Shuttle',	'C-MAX',	'C 240',	'Land Cruiser Prado',	'328 Xdrive',	'E 240',	'Taurus',	'Twingo',	'535 M PAKET',	'S 500 long',	'520',	'525',	'A4 premium',	'Galaxy',	
                                                            'Kizashi',	'Golf 4',	'Element',	'C1',	'X-Trail',	'RIO',	'Discovery',	'Vento',	'C 200 2.0',	'325',	'Corolla IM',	'CLK 240',	'Fiesta',	'Cooper',	'Combo',	'Challenger',	'RX 300',	'A4',	'320',	'RX 400 HYBRID',	'Corolla verso',	'Pajero IO',	'528 i',	'Odyssey',	'L 200',	'Insight EX',	'Fiesta 1.6',	'Focus',	'Teana',	'Matiz',	'Mazda 3',	'Corsa',	'Scenic',	'NX 200',	'Spark',	'March',	'HS 250h Hybrid',	'Journey',	'Sonata SPORT',	'Elantra SE',	'Rogue Sport',	'Auris',	'FJ Cruiser',	'ES 350',	'Impala',	
                                                            'SOUL',	'500C Lounge',	'X-Terra',	'Montero',	'Cruze ltz',	'X5 x5',	'Ipsum',	'Countryman',	'Corolla 04',	'A 190',	'SLK 230',	'Rogue SL',	'Q3',	'Accent',	'Grandeur',	'Stella',	'Hr-v',	'Prius 2014',	'Dart Limited',	'CX-9',	'200',	'Accord',	'Compass',	'Transit Connect',	'XV',	'Aqua g soft leather sele',	'Meriva',	'Prius V',	'Sorento',	'114',	'RX 400',	'Passo',	'ES 300',	'Sportage',	'320 DIESEL',	'318 áƒ¡áƒáƒ¡áƒ¬áƒ áƒáƒ¤áƒáƒ“',	'A5',	'Versa',	'FIT Sport',	'Carnival',	'Step Wagon Pada',	'Dart GT 2.4',	'TLX',	'E 230',	'A3 PREMIUM',	
                                                            'GL 320',	'Tourneo Connect',	'kona',	'335',	'CLK 320 AMG',	'730 3.0',	'ISIS',	'1300',	'Sprinter 411',	'Sintra',	'E 500',	'X5 M',	'CLS 350',	'Transit 350T',	'435',	'Outlander 2.0',	'Polo',	'Cherokee',	'CLA 250',	'Ist 1.5',	'2107',	'Note',	'Harrier',	'XF',	'X6 M',	'CLS 550',	'Octavia',	'Mazda 3 SPORT',	'IX35',	'Tiida 2008',	'Sentra',	'Town Car',	'Mirage',	'Frontier',	'325 CI',	'Smart',	'Panamera',	'Jetta áƒ¡áƒžáƒáƒ áƒ¢',	'Corolla LE',	'428',	'Transit Connect áƒ‘áƒ”áƒœáƒ–áƒ˜áƒœáƒ˜',	'Elantra limited',	'Camry S',	'Ipsum S',	'Cooper S Cabrio R56',	'LS 460',	
                                                            'Mx-5',	'Crafter',	'Prius plugin',	'A 140',	'Cadenza',	'Sonata 2.0t',	'E 280',	'Sonata S',	'GL 450',	'B 170',	'Cerato',	'Edge',	'PT Cruiser',	'Astra 1600',	'Sonata blue edition',	'CLS 55 AMG',	'Sprinter VAN',	'CLK 430',	'IS 350',	'Civic EX',	'Fuga',	'Viano',	'MKZ',	'528',	'Cruze strocna',	'Countryman S',	'Tiida AXIS',	'CLK 200',	'Swift',	'Volt premier',	'HS 250h',	'CT 200h F-sport',	'Mondeo',	'316 i',	'Century',	'E 550',	'Legacy B4 twin turbo',	'IS 250',	'740',	'A8',	'Escort',	'Tiguan',	'Astra BERTONE',	'FIT Hbrid',	'Verso',	'S 550',	'Golf 6',	'Bluebird',	'120',	'CLK 230 .',	
                                                            'Liana',	'C 280',	'Alphard',	'Passport',	'ColtPlus',	'A4 S line',	'Camry XLE',	'Sonata hybrid',	'Sierra',	'Liberty',	'Fit Aria',	'F-type',	'Grandis',	'E 430',	'FIT S',	'Inspire',	'Venza',	'RVR',	'Town and Country',	'GL 550',	'XV LIMITED',	'Premacy',	'R 350',	'135',	'Corolla S',	'Traverse',	'Demio',	'Jetta GLI',	'CLK 55 AMG',	'CX-5',	'C 230',	'Combo 2001',	'S 500',	'Fusion phev',	'500 Abarth',	'650',	'206',	'A4 premium plius',	'Mazda 6',	'Fred',	'Altezza',	'C 320 CDI',	'2106',	'550',	'E 270',	'Escape Titanium',	'C30 2010',	'Micra',	'X5 XDRIVE',	'535 i',	'ML 280 áƒ¡áƒáƒ¡áƒ¬áƒ áƒáƒ¤áƒáƒ“',	'Camry SPORT',	'Impreza',	'Hilux',	'Cruze Premier',	
                                                            'UP',	'Outlander SPORT',	'500L LONG',	'Ranger',	'XJ',	'XL7',	'Passat sel',	'FIT HIBRID',	'CC R line',	'Mazda 6 TOURING',	'Durango',	'Caliber',	'Murano',	'Transit CL',	'523',	'320 i',	'Escape 3.0',	'Ramcharger',	'Transit Fff',	'LX 570',	'Gentra',	'316',	'Cinquecento',	'Pilot',	'5.3E+62',	'CC 2.0 T',	'BRZ',	'SX4',	'Touareg',	'1500',	'Grand Cherokee Saiubileo',	'Smart Fortwo',	'Skyline',	'500 Sport',	'Golf TDI',	'Demio evropuli',	'X3',	'Equinox LT',	'Navigator',	'Outlander SE',	'E 270 AVANGARDI',	'GS 350',	'Wingroad',	'Passat R-line',	'Eunos 500',	'Doblo',	'Allroad',	'E 350 212',	'Freelander',	'Serena Serea',	'Caddy',	'E 55',	'Cooper S Cabrio',	'Transit S',	'C-MAX HYBRID',	'ML 500',
                                                                	'ML 270',	'CLK 200 Kompressor',	'Samurai',	'M5 Japan',	'Caldina',	'607',	'Astra H',	'Megane GT Line',	'CLS 350 AMG',	'Q5 S-line',	'X1',	'CLK 270',	'RDX',	'Elantra GT',	'Cruze RS',	'2121 (Niva)',	'MKZ hybrid',	'130',	'X-Trail X-trail',	'Hilux Surf',	'CC',	'Prius C Navigation',	'E 350 w211',	'Expedition',	'ML 320',	'Sprinter 315CDI',	'Taurus interceptor',	'RIO lx',	'X5 XDRIVE 35D',	'Jetta SEL',	'Wrangler',	'G 300',	'RX 450 H',	'Agila',	'Highlander sport',	'Avella',	'Focus TITANIUM',	'Acadia',	'Forte',	'Accord CL9 type S',	'500 turbo',	'TSX',	'Aqua L paketi',	'Santa FE Ultimate',	'Sprinter 311',	'320 Gran Turismo',	'E 280 CDI',	'Astra astra',	'311',	'535 Twinturbo',	'416',	'Avensis',	'Sonata Hibrid',	'Prius s',	'Will Vs',	'Prius BLUG-IN',	'Presage RIDER',	'Yaris IA',	'Camaro',	'X5 X-Drive',	'Getz',	'500C',	'R 320',	'MDX',	'KA',	'128 M tech',	'Encore',	'Megane',	'FIT HYBRYD',	'CL 550',	'X3 3.5i',	'E 350 AMG',	'Estima',	'C 220 CDI',	'550 F10',	'Focus ST',	'Camry sporti',	'Sonic',	'545',	'Elysion',	'X5 4,4i',	'320 2.2',	'Megane 19',	'S 320',	'Cruze LS',	'Camry HYBRID',	'E 350 4 Matic AMG Packag',	'645 CI',	'CTS',	'Camry LE',	'CT 200h F SPORT',	'Maverick',	'225',	'S-max',	'525 i',	'GL 320 bluetec',	'100',	'Quest 2016',	'RC F',	'320 2.0',	'328 DIZEL',	'Impreza G4',	'FIT fit',	'RS7',	'Passat SE',	'iA isti',	'Elantra GLS / LIMITED',	'Optima X',	'C 320',	'Touran',	'Omega B',	'Corsa Corsa',	'Vectra C',	'E 220 cdi',	'Demio 12',	'535 comfort-sport',	'500 46 ml',	'Wish',	'Vitz RS',	'500 Lounge',	'XC90',	'Ioniq',	'525 525',	'Trailblazer',	'Optima Hybrid',	'G 320',	'Crosstrek',	'R 350 BLUETEC',	'Cruze sonic',	'Nubira',	'Axela',	'X5 3.0',	'Scirocco',	'745 i',	'A3',	'SRX',	'Renegade',	'Caravan',	'S 350 Longia',	'Fun Cargo',	'Cruze L T',	'C-MAX SEL',	'Vitara',	'Aqua G klas',	'735',	'Cougar',	'Fusion SE',	'NEW Beetle',	'116',	'328 sulev',	'Sorento SX',	'4Runner',	'Kangoo',	'Aerio SX',	'G 55 AMG',	'C 230 2.5',	'Omega',	'CERVO',	'Sprinter Maxi-áƒ¡ Max',	'Sonata LPG',	'Kizashi sporti',	'B-MAX',	'Montero Sport',	'X1 28Xdrive',	'Malibu LT',	'X5 restilling',	'Focus Fokusi',	'Fabia',	'IS 250 áƒ áƒ”áƒ¡áƒ¢áƒáƒ˜áƒšáƒ˜áƒœáƒ’áƒ˜',	'335 335i',	'IS 200',	'tC',	'Intrepid',	'Prius áƒ¤áƒšáƒáƒ’áƒ˜áƒœáƒ˜',	'CX-5 Touring',	'Edix',	'ML 350 BLUETEC',	'IX35 2.0',	'750 4.8',	'Veloster R-spec',	'Azera',	'2107 07',	'Transit Custom',	'960',	'Elysion 3.0',	'Zafira B',	'Prius 1.5I',	'ML 270 CDI',	'Civic Ferio',	'Carisma',	'S60',	'Jetta TDI',	'Optima ECO',	'C 200 KOMPRESSOR',	'X5 rest',	'Escudo',	'Verisa',	'Kicks',	'ATS',	'Mark X',	'Vitara GL+',	'X1 X-Drive',	'Focus Flexfuel',	'Aqua G',	'Swift Sport',	'Patriot',	'300',	'Duster',	'Q7 sport',	'Transit Connect Prastoi',	'Verisa 2007',	'Camry sport se',	'ML 280',	'Highlander LIMITED',	'Grand Cherokee LAREDO',	'E 270 CDI',	'CLK 230',	'X5 3.0i',	'Corolla spacio',	'Camry XV50',	'3.25E+48',	'911 meqanika',	'645',	'Transit T330',	'Prius áƒ°áƒ˜áƒ‘áƒ áƒ˜áƒ“áƒ˜',	'Escape SE',	'3.2E+38',	'Combo TDI',	'E 50',	'Aqua HIBRID',	'Celica',	'ML 350 3.7',	'Escape áƒ›áƒ”áƒ áƒ™áƒ£áƒ áƒ˜ áƒ›áƒ”áƒ áƒ˜áƒœáƒ”áƒ áƒ˜',	'macan',	'Panda',	'Niro',	'X-Trail gt',	'C 400',	'RAV 4 SPORT',	'CR-Z',	'Sprinter áƒ¡áƒáƒ¢áƒ•áƒ˜áƒ áƒ—áƒ',	'Malibu eco',	'Forester stb',	'EcoSport SE',	'FIT Premiym',	'Legacy Bl5',	'HHR',	'Prius TSS LIMITED',	'Cooper r50',	'C8',	'Cr-v Cr-v',	'A3 4X4',	'100 NX',	'M3',	'Land Cruiser',	'Vitz i.ll',	'T3',	'RAV 4 XLE',	'Golf 2',	'GLK 300',	'Golf 3',	'Range Rover',	'E 420',	'270',	'C 250 luxury',	'Passat sport',	'E 320 4Ã—4',	'Jetta se',	'500L',	'Ghibli',	'C-MAX C-MAX',	'X1 4X4',	'530 GT',	'Land Cruiser Prado RX',	'GLK 350',	'M6',	'325 i',	'190',	'C 300 4matic',	'207',	'i3',	'FIT RS MODELI',	'C 250 1.8 áƒ¢áƒ£áƒ áƒ‘áƒ',	'Prius C 2013',	'C4',	'Kicks SR',	'Caliber sxt',	'E-pace',	'X-type',	'Cefiro',	'Avalanche',	'CR-Z áƒ°áƒ˜áƒ‘áƒ áƒ˜áƒ“áƒ˜',	'M5',	'A6 Ð¡6',	'C-MAX PREMIUM',	'Tigra',	'A 140 140',	'328 DRIFT CAR',	'Forester XT',	'GX 470 470',	'March 231212',	'E 400',	'Cruze Cruze',	'328 i',	'ML 350 4matic',	'Eos',	'Astra td',	'Tucson Limited',	'Ignis',	'Camaro LS',	'Corolla ECO',	'Catera',	'Cayman',	'Land Rover Sport',	'E 350 4 MATIC',	'Golf Gti',	'Dart',	'Demio mazda2',	'745',	'Neon',	'A4 B6',	'Astra GE',	'Octavia Scout',	'Sprinter VIP CLASS',	'A6 UNIVERSAL',	'Mariner',	'Mustang cabrio',	'CLK 280',	'FIT RS',	'Avenger',	'500 s',	'Eclipse',	'Move',	'C5',	'3.18E+38',	'C 230 kompresor',	'Vanette',	'Q5 Prestige',	'Jetta 2.0',	'Hiace',	'S 430',	'Discovery LR3',	'Atenza',	'Citan',	'ColtPlus Plus',	'Passat RLAINI',	'Alto Lapin',	'Outlander áƒ¡áƒžáƒáƒ áƒ¢',	'Quest',	'Countryman S turbo',	'Prius V HYBRID',	'Malibu Hybrid',	'B 220',	'E 320 4matic',	'S 55 5.5',	'GS 300',	'Camry sel',	'GLE 450',	'Elantra 2014',	'520 I',	'Belta',	'CLK 200 200',	'Transit Tourneo',	'Trax',	'C-MAX SE',	'Demio Sport',	'IS 250 TURBO',	'280',	'F150',	'CX-3',	'525 ///M',	'Volt Full Packet',	'Fred HIBRIDI',	'Forester 4x4',	'Juke Nismo RS',	'E 270 4',	'Phaeton',	'Millenia',	'Lancer GT',	'250',	'Prius 9',	'ML 320 cdi',	'A6 QUATTRO',	'730',	'318 m',	'Continental',	'March Rafeet',	'E 200 w210',	'318 áƒ áƒ”áƒ¡áƒ¢áƒáƒ˜áƒšáƒ˜áƒœáƒ’áƒ˜',	'Juke Nismo',	'F-pace',	'Range Rover VOGUE',	'Lantra',	'X3 SDRIVE',	'FIT RS MUGEN',	'C70',	'JX35',	'Forester CrossSport',	'GL 350 BLUTEC',	'XL7 limited',	'Outback Limited',	'A4 B5',	'X5 Japan',	'CRX',	'C 250 A.M.G',	'XC90 3.2 AWD',	'535 535',	'550 M Packet',	'E 250',	'Transit 100LD',	'Crossland X',	'Forester cross sport',	'Maxima',	'Chariot',	'GL 350',	'Grand HIACE',	'Passat pasat',	'230',	'C 250 AMG',	'Gloria',	'C 180 komp',	'Yaris RS',	'CL 500',	'118',	'Sportage SX',	'X-Trail NISSAN X TRAIL R',	'Prius prius',	'Astra suzuki mr wagon',	'Pajero MONTERO',	'Range Rover Evoque 2.0',	'Sonata Limited',	'S80',	'Tucson Se',	'Sprinter 313CDI',	'Camry XLEi',	'Captur QM3 Samsung',	'300 LIMITED',	'Passat 2.0 tfsi',	'Jetta SPORT',	'Sonata 2.4L',	'NX 300',	'LATIO',	'CC sport',	'Transit áƒžáƒ”áƒ áƒ”áƒ’áƒáƒ áƒáƒ¢áƒ™áƒ',	'Pajero 2.5diezel',	'ML 350 370',	'406',	'Prius 11',	'370Z',	'Explorer Turbo japan',	'Fusion Bybrid',	'Vesta',	'Lupo iaponuri',	'118 2,0',	'Transporter',	'C 180 2.0',	'Land Cruiser 105',	'Golf 1.8',	'X-Trail NISMO',	'Step Wagon',	'Lantra LIMITED',	'Frontera',	'Land Cruiser 100',	'Veloster Turbo',	'E 320 bluetec',	'SJ 413 Samurai',	'Vito 115 CDI',	'Routan SEL',	'Grand Cherokee special e',	'335 D',	'Vectra áƒ‘',	'220',	'ML 350 ML350',	'R2',	'Pathfinder SE',	'500 Abarth áƒ¢áƒ£áƒ áƒ‘áƒ',	'Matrix XR',	'Presage',	'Mazda 2',	'Edix FR-v',	'535 XI',	'Prius plug-in',	'RAV 4 Dizel',	'IS-F',	'Astra j',	'A 200',	'Kyron',	'Prius personna',	'S3',	'Fusion 2015',	'Sportage EX',	'Omega c',	'Optima EX',	'Navara',	'Legacy Outback',	'FIT Modulo',	'626',	'RX 450 HYBRID',	'Astra CNG',	'Prius C áƒ°áƒ˜áƒ‘áƒ áƒ˜áƒ“áƒ˜',	'ML 350 sport',	'Noah',	'Mazda 6 Grand touring',	'Sorento EX',	'Camry SPORT PAKET',	'SLK 32 AMG',	'C 32 AMG',	'Berlingo',	'Fusion HIBRID',	'S 550 LONG',	'Sprinter 316',	'525 TDI',	'FIT GP-5',	'X5 M packet',	'Kangoo Waggon',	'Legacy B4',	'Ceed',	'307',	'500X',	'Sonic LT',	'H1 grandstarex',	'Sportage PRESTIGE',	'Skyline 4WD',	'Will Chypa',	'C 320 AMG',	'S40',	'RAM 1500',	'T3 0000',	'Superb',	'Megane 1.9CDI',	'E 220 W210...CDI',	'S 600',	'Jetta áƒ¡áƒáƒ¡áƒ¬áƒáƒ¤áƒáƒ“',	'Corvette',	'GLK 250',	'Sprinter 316 CDI',	'E 36 AMG',	'Outlander xl',	'FIT "S"- PAKETI.',	'REXTON SUPER',	'Fiesta SE',	'H1 GRAND STAREX',	'ML 63 AMG',	'Cayenne S',	'Tiida Latio',	'Wrangler sport',	'ML 55 AMG',	'Sai',	'500X Lounge',	'M550',	'Qashqai Advance CVT',	'Accent SE',	'535 i xDrive',	'Countryman sport',	'Sprinter 313',	'ES 300 hybrid',	'Tribute',	'Optima SXL',	'Caravan tradesman',	'Almera',	'FIT ex',	'128',	'Mariner Hybrid',	'Vito long115',	'GL 350 áƒ“áƒ˜áƒ–áƒ”áƒšáƒ˜',	'VOXY 2003',	'Z4',	'Vito 113',	'A 170 CDI',	'Ractis',	'530 M',	'Pajero Mini 2008 áƒ¬áƒšáƒ˜áƒáƒœáƒ˜',	'E 280 3.0',	'Eclipse ES',	'Fusion 1.6',	'Transit ford',	'Passat B7',	'Lupo',	'Carens',	'Quattroporte',	'Elantra 2016',	'Crossroad',	'A4 S4',	'Aqua áƒ¡áƒáƒ¡áƒ¬áƒ áƒáƒ¤áƒáƒ“',	'M4',	'Veloster remix',	'Rasheen',	'A4 B7',	'Seicento fiat 600',	'C 270',	'508',	'Vito 111 CDI',	'xD',	'EcoSport',	'340',	'Passat B5',	'Polo GTI 16V',	'XV HYBRID',	'323 F',	'Elantra LIMITEDI',	'Sienta LE',	'One',	'TT',	'2109',	'ML 500 AMG',	'Space Runner',	'Qashqai SPORT',	'S 350 CDI 320',	'Astra GTC 1.9 turbo dies',	'400X',	'C1 C',	'Moco',	'Monterey',	'3008',	'2101 01',	'B 200',	'Frontera A B',	'C 240 w203',	'Vectra H',	'CLA 250 AMG',	'Highlander 2,4',	'PT Cruiser pt cruiser',	'807',	'Prius C 80 original',	'Insight LX',	'Patriot 70th anniversary',	'CLK 320 avangarde',	'Scorpio',	'Airtrek turbo',	'A6 premium plus',	'Discovery IV',	'Tribute áƒ¡áƒáƒ¡áƒ¬áƒ áƒáƒ¤áƒáƒ“',	'318 318',	'Silvia',	'Astra gi',	'Sprinter 315CDI-XL',	'GL 500',	'FIT GP-6',	'530 525i',	'IS 350 C',	'B 200 Turbo',	'CLK 200 kompresor',	'Megane 5',	'C 300 sport',	'Elantra i30',	'Prius Plug IN',	'G 230 2.2cdi',	'X5 DIESEL',	'DS 4',	'520 d xDrive Luxury',	'Vectra 1.6',	'Hr-v EXL',	'Mark X Zio',	'Aqua sport',	'Almera dci',	'Impreza Sport',	'X4',	'Bora',	'Fusion hybrid',	'Vito 111',	'2105',	'Daimler',	'Jetta sei',	'Sprinter 308 CDI',	'TL saber',	'Courier',	'Sprinter EURO4',	'MPV LX',	'Yaris SE',	'Viano Ambiente',	'118 M-sport LCI',	'CT 200h 1.8',	'Envoy',	'Combo 1700',	'Passat tdi sel',	'B 180',	'Land Cruiser 200',	'Santa FE sport',	'C 250 1,8 turbo',	'Highlander XLE',	'Vito 115',	'Sonata SE LIMITED',	'Prius 1.8',	'208',	'H1 starixs',	'Skyline GT250',	'Sambar',	'Accent GS',	'Corolla se',	'A5 Sportback',	'Mazda 5',	'Charger RT',	'C 250 AMG-PAKET-1,8',	'525 Vanos',	'Primera',	'Niva',	'7.3E+34',	'335 M paket',	'B 170 B Class',	'B9 Tribeca',	'206 CC',	'S 430 4.3',	'E 220 211',	'C 250 1.8',	'Megane 1.9 CDI',	'X5 E70',	'LAFESTA',	'XK',	'S 420',	'Regal',	'ML 350 SPECIAL EDITION',	'E 260',	'Prius C hybrid',	'Juke juke',	'FIT Hybrid',	'Cooper F-56',	'G37',	'2103 03',	'Galant',	
                                                                    'ML 300',	'V50',	'Caliber journey',	'CLK 200 208',	'Terrano',	'V 230',	'GS 450',	'407',	'CLK 350',	
                                                                    'Camry XSE',	'500 sport panorama',	'B 170 Edition One',	'Crafter 2,5TDI',	'Escalade',	'740 i',	'Galant GTS',	'FIT PREMIUMI',	'Elantra GS',	'DTS',	'Optima hybid',	'Sierra DIZEL',	'Passat tsi-se',	'Caddy cadi',	'Paceman',	'Taurus X',	'Camaro RS',	'Rx-8',	'E 290',	'535 M',	'C 230 2.0 kompresor',	'Micra <DIESEL>',	'i20',	'Prius plagin',	'Punto',	'Prius 3',	'RAV 4 L',	'335 áƒ¢áƒ£áƒ áƒ‘áƒ',	'BB',	'Camry áƒ°áƒ˜áƒ‘áƒ áƒ˜áƒ“áƒ˜',	'Crosstour',	'Camry SE HIBRYD',	'428 i',	'Juke Turbo',	'i40',	'Versa s',	'Legend FULL',	'FIT PREMIUM PAKETI',	'Lancer GTS',	'530 I',	'Jetta s',	'Cooper CLUBMAN',	'E 200 CGI',	'Jetta 2',	'Golf GOLF 5',	'E 200 2000',	'Integra',	'Impreza WRX/STI LIMITED',	'CL 55 AMG KOMPRESSOR',	'Delica 5',	'Corolla 140',	'Cooper S',	'Pajero Mini 2010 áƒ¬áƒšáƒ˜áƒáƒœáƒ˜',	'Cruze LT RS',	'2111',	'316 1995',	'A6 C7',	'Astra A.H',	'HUSTLER',	'Step Wagon RG2 SPADA',	'Jetta Hybrid',	'FIT NAVI PREMIUM',	'Civic Hybrid',	'Escape áƒ¡áƒáƒ¡áƒ¬áƒ áƒáƒ¤áƒáƒ“',	'Z4 3,0 SI',	'Jetta 1.4 TURBO',	'Forester L.L.BEAN',	'Sonata SE',	'S 500 67',	'INSIGNIA',	'ML 320 AMG',	'Escape escape',	'Veracruz',	'940',	'Galloper',	'Transit 2.4',	'325 XI',	'Sonata áƒ¡áƒáƒ¡áƒ¬áƒ áƒáƒ¤áƒáƒ“',	'Cami',	'T5',	'Tiguan SE',	'A6 evropuli',	'Prius C YARIS IA',	'CL550 AMG',	'Outback 3.0',	'A 170 Avangard',	'Prius C 1.5I',	'Patriot Latitude',	'Kalos',	'A4 Sline',	'ML 550',	'Ridgeline',	'Prius V HIBRID',	'XC90 2.5turbo',	'Focus SEL',	'X5 35d',	'Cruze S',	'SLK 350 300',	'RX 400 H',	'Optima k5',	'X5 Sport',	'Minica',	'528 3.0',	'Outback 2007',	'Fusion HYBRID SE',	'Versa SE',	'Vito Exstralong',	'C 240 W 203',	'S70',	'CLS 63 AMG',	
                                                                    'S-type',	'Vito Extralong',	'FIT LX',	'Every Landy NISSAN SEREN',	'QX56',	'E 230 124',	'Prius C aqua'])

        Manufacturer = st.selectbox(label="Select the Manufacturer", options=['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN', 'AUDI', 'RENAULT', 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA', 'MITSUBISHI', 'SSANGYONG', 
                                                                              'MAZDA', 'GMC', 'FIAT', 'INFINITI', 'SUZUKI', 'ACURA', 'VAZ', 'CITROEN', 'LAND ROVER',
                                                                                'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR', 'SKODA', 'LINCOLN', 'BUICK', 'PEUGEOT', 'VOLVO', 'CADILLAC', 
                                                                                'SCION', 'MERCURY', 'MASERATI', 'DAIHATSU'])

        color = st.selectbox(label="Select the color of the car", options=['Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 'Sky blue', 'Red', 'Yellow', 'Brown', 'Golden', 'Beige', 'Orange', 'Carnelian red', 'Purple', 'Pink'])
        Category = st.selectbox(label="Select the type of the car", options=['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Pickup', 'Limousine'])
        Fuel_type = st.selectbox(label="Select Fuel Type", options=['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG', 'Hydrogen'])
        Gear_box = st.selectbox(label="Type of Gear Transmission", options=['Automatic', 'Tiptronic', 'Variator', 'Manual'])
        Leather_interior = st.selectbox(label="Does the car have leather interior?", options=[True, False])
        airbags = st.selectbox('Number of Airbags', options=range(1, 16))
        #Engine_volume = st.selectbox('Engine Volume', options=range(0.1, 7.0))
        Engine_volume = st.number_input('Engine Volume', min_value=0.1, max_value=7.0, value=0.1, step=0.1)
        prod_year = st.selectbox('Year of Manufacture', options=range(1930, 2023))
        #Levy = st.selectbox('Enter Levy', options=range(150.0, 1700.0))
        #Levy = st.slider('Enter Levy', min_value=150.0, max_value=1700.0, step=0.1)
        Levy = st.number_input('Enter Levy', min_value=150.0, max_value=1700.0,value=150.0, step=0.1)
        #Mileage = st.selectbox('Enter Mileage', options=range(0, 367000))
        Mileage = st.number_input('Enter Mileage', min_value=0, max_value=377000,value=20000, step=10000)
        
        # Add submit button to form
        submitted = st.form_submit_button(label='Submit')
        
        # If user submits form
        if submitted:
            # Create a dictionary from the user input and convert it into a DataFrame
            my_data = {'Model': [Model],
                       'Manufacturer': [Manufacturer],
                       'color': [color],
                       'Category': [Category],
                       'Fuel_type': [Fuel_type],
                       'Gear_box': [Gear_box],
                       'Leather_interior': [1] if Leather_interior == True else [0],
                       'airbags': [airbags],
                       'Engine_volume': [Engine_volume],
                       'prod_year': [prod_year],
                       'Levy': [Levy],
                       'Mileage': [Mileage]
                       }
            
            my_df = pd.DataFrame(my_data)
            
            # Preprocess the input data using the preprocessor pipeline
            full_pipeline.fit_transform(my_df)
            my_df_processed = full_pipeline.transform(my_df)
            # Scale the input data using MinMax Scaler
            #scaler = StandardScaler() 
            #my_df_scaled = pd.DataFrame(scaler.fit_transform(my_df), columns=my_df.columns)
            
            # Make prediction using LRM model
            my_y_pred = RFR.predict(my_df_processed)
        
            # Display prediction result
            st.write('The estimated price of the car is:', my_y_pred[0])

if __name__ == '__main__':
    main()


