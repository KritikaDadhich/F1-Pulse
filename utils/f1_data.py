"""
f1_data.py
Static F1 historical data: race results, driver standings, team colors, stock links.
Separated from app.py so it can be loaded once via @st.cache_data instead of being
re-parsed as a Python literal on every Streamlit script rerun.
"""

ALL_RACES = {
    2021: [
        {"round":1,"race":"Bahrain GP","circuit":"Bahrain International Circuit","date":"2021-03-28","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Verstappen","p3":"Bottas"},
        {"round":2,"race":"Emilia Romagna GP","circuit":"Imola","date":"2021-04-18","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Bottas"},
        {"round":3,"race":"Portuguese GP","circuit":"Portimao","date":"2021-05-02","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Verstappen","p3":"Bottas"},
        {"round":4,"race":"Spanish GP","circuit":"Circuit de Barcelona-Catalunya","date":"2021-05-09","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Verstappen","p3":"Bottas"},
        {"round":5,"race":"Monaco GP","circuit":"Circuit de Monaco","date":"2021-05-23","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Sainz","p3":"Leclerc"},
        {"round":6,"race":"Azerbaijan GP","circuit":"Baku City Circuit","date":"2021-06-06","winner":"Sergio Perez","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Vettel","p3":"Gasly"},
        {"round":7,"race":"French GP","circuit":"Circuit Paul Ricard","date":"2021-06-20","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Bottas"},
        {"round":8,"race":"Styrian GP","circuit":"Red Bull Ring","date":"2021-06-27","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Bottas"},
        {"round":9,"race":"Austrian GP","circuit":"Red Bull Ring","date":"2021-07-04","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Bottas"},
        {"round":10,"race":"British GP","circuit":"Silverstone Circuit","date":"2021-07-18","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Leclerc","p3":"Bottas"},
        {"round":11,"race":"Hungarian GP","circuit":"Hungaroring","date":"2021-08-01","winner":"Esteban Ocon","winner_team":"Alpine","event_type":"WIN","p2":"Hamilton","p3":"Vettel"},
        {"round":12,"race":"Belgian GP","circuit":"Spa-Francorchamps","date":"2021-08-29","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Russell","p3":"Hamilton"},
        {"round":13,"race":"Dutch GP","circuit":"Circuit Zandvoort","date":"2021-09-05","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Alonso","p3":"Hamilton"},
        {"round":14,"race":"Italian GP","circuit":"Autodromo Nazionale Monza","date":"2021-09-12","winner":"Daniel Ricciardo","winner_team":"McLaren","event_type":"WIN","p2":"Norris","p3":"Bottas"},
        {"round":15,"race":"Russian GP","circuit":"Sochi Autodrom","date":"2021-09-26","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Verstappen","p3":"Sainz"},
        {"round":16,"race":"Turkish GP","circuit":"Istanbul Park","date":"2021-10-10","winner":"Valtteri Bottas","winner_team":"Mercedes","event_type":"WIN","p2":"Verstappen","p3":"Leclerc"},
        {"round":17,"race":"US GP","circuit":"Circuit of the Americas","date":"2021-10-24","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Bottas"},
        {"round":18,"race":"Mexico City GP","circuit":"Autodromo Hermanos Rodriguez","date":"2021-11-07","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Bottas"},
        {"round":19,"race":"São Paulo GP","circuit":"Autodromo Jose Carlos Pace","date":"2021-11-14","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Verstappen","p3":"Bottas"},
        {"round":20,"race":"Qatar GP","circuit":"Lusail International Circuit","date":"2021-11-21","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Alonso","p3":"Gasly"},
        {"round":21,"race":"Saudi Arabian GP","circuit":"Jeddah Corniche Circuit","date":"2021-12-05","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Verstappen","p3":"Bottas"},
        {"round":22,"race":"Abu Dhabi GP","circuit":"Yas Marina Circuit","date":"2021-12-12","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Leclerc"},
    ],
    2022: [
        {"round":1,"race":"Bahrain GP","circuit":"Bahrain International Circuit","date":"2022-03-20","winner":"Charles Leclerc","winner_team":"Ferrari","event_type":"WIN","p2":"Sainz","p3":"Hamilton"},
        {"round":2,"race":"Saudi Arabian GP","circuit":"Jeddah Corniche Circuit","date":"2022-03-27","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Carlos Sainz"},
        {"round":3,"race":"Australian GP","circuit":"Albert Park Circuit","date":"2022-04-10","winner":"Charles Leclerc","winner_team":"Ferrari","event_type":"WIN","p2":"Perez","p3":"Colapinto"},
        {"round":4,"race":"Emilia Romagna GP","circuit":"Imola","date":"2022-04-24","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Perez","p3":"Norris"},
        {"round":5,"race":"Miami GP","circuit":"Miami International Autodrome","date":"2022-05-08","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Sainz"},
        {"round":6,"race":"Spanish GP","circuit":"Circuit de Barcelona-Catalunya","date":"2022-05-22","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Perez","p3":"Russell"},
        {"round":7,"race":"Monaco GP","circuit":"Circuit de Monaco","date":"2022-05-29","winner":"Sergio Perez","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Sainz","p3":"Verstappen"},
        {"round":8,"race":"Azerbaijan GP","circuit":"Baku City Circuit","date":"2022-06-12","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Sainz","p3":"Perez"},
        {"round":9,"race":"Canadian GP","circuit":"Circuit Gilles Villeneuve","date":"2022-06-19","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Sainz","p3":"Hamilton"},
        {"round":10,"race":"British GP","circuit":"Silverstone Circuit","date":"2022-07-03","winner":"Carlos Sainz","winner_team":"Ferrari","event_type":"WIN","p2":"Perez","p3":"Leclerc"},
        {"round":11,"race":"Austrian GP","circuit":"Red Bull Ring","date":"2022-07-10","winner":"Charles Leclerc","winner_team":"Ferrari","event_type":"WIN","p2":"Verstappen","p3":"Hamilton"},
        {"round":12,"race":"French GP","circuit":"Circuit Paul Ricard","date":"2022-07-24","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Hamilton"},
        {"round":13,"race":"Hungarian GP","circuit":"Hungaroring","date":"2022-07-31","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Russell"},
        {"round":14,"race":"Belgian GP","circuit":"Spa-Francorchamps","date":"2022-08-28","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Perez","p3":"Sainz"},
        {"round":15,"race":"Dutch GP","circuit":"Circuit Zandvoort","date":"2022-09-04","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Perez"},
        {"round":16,"race":"Italian GP","circuit":"Autodromo Nazionale Monza","date":"2022-09-11","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Russell"},
        {"round":17,"race":"Singapore GP","circuit":"Marina Bay Street Circuit","date":"2022-10-02","winner":"Sergio Perez","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Sainz"},
        {"round":18,"race":"Japanese GP","circuit":"Suzuka Circuit","date":"2022-10-09","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Perez","p3":"Leclerc"},
        {"round":19,"race":"US GP","circuit":"Circuit of the Americas","date":"2022-10-23","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Leclerc"},
        {"round":20,"race":"Mexico City GP","circuit":"Autodromo Hermanos Rodriguez","date":"2022-10-30","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Russell"},
        {"round":21,"race":"São Paulo GP","circuit":"Autodromo Jose Carlos Pace","date":"2022-11-13","winner":"George Russell","winner_team":"Mercedes","event_type":"WIN","p2":"Hamilton","p3":"Verstappen"},
        {"round":22,"race":"Abu Dhabi GP","circuit":"Yas Marina Circuit","date":"2022-11-20","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Hamilton"},
    ],
    2023: [
        {"round":1,"race":"Bahrain GP","circuit":"Bahrain International Circuit","date":"2023-03-05","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Perez","p3":"Alonso"},
        {"round":2,"race":"Saudi Arabian GP","circuit":"Jeddah Corniche Circuit","date":"2023-03-19","winner":"Sergio Perez","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Alonso","p3":"Leclerc"},
        {"round":3,"race":"Australian GP","circuit":"Albert Park Circuit","date":"2023-04-02","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Alonso"},
        {"round":4,"race":"Azerbaijan GP","circuit":"Baku City Circuit","date":"2023-04-30","winner":"Sergio Perez","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Verstappen","p3":"Leclerc"},
        {"round":5,"race":"Miami GP","circuit":"Miami International Autodrome","date":"2023-05-07","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Alonso","p3":"Leclerc"},
        {"round":6,"race":"Monaco GP","circuit":"Circuit de Monaco","date":"2023-05-28","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Alonso","p3":"Leclerc"},
        {"round":7,"race":"Spanish GP","circuit":"Circuit de Barcelona-Catalunya","date":"2023-06-04","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Russell"},
        {"round":8,"race":"Canadian GP","circuit":"Circuit Gilles Villeneuve","date":"2023-06-18","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Alonso","p3":"Hamilton"},
        {"round":9,"race":"Austrian GP","circuit":"Red Bull Ring","date":"2023-07-02","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Perez"},
        {"round":10,"race":"British GP","circuit":"Silverstone Circuit","date":"2023-07-09","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Hamilton"},
        {"round":11,"race":"Hungarian GP","circuit":"Hungaroring","date":"2023-07-23","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Hamilton"},
        {"round":12,"race":"Belgian GP","circuit":"Spa-Francorchamps","date":"2023-07-30","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Perez","p3":"Hamilton"},
        {"round":13,"race":"Dutch GP","circuit":"Circuit Zandvoort","date":"2023-08-27","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Albon","p3":"Hamilton"},
        {"round":14,"race":"Italian GP","circuit":"Autodromo Nazionale Monza","date":"2023-09-03","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Sainz","p3":"Leclerc"},
        {"round":15,"race":"Singapore GP","circuit":"Marina Bay Street Circuit","date":"2023-09-17","winner":"Carlos Sainz","winner_team":"Ferrari","event_type":"WIN","p2":"Norris","p3":"Hamilton"},
        {"round":16,"race":"Japanese GP","circuit":"Suzuka Circuit","date":"2023-09-24","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Norris"},
        {"round":17,"race":"Qatar GP","circuit":"Lusail International Circuit","date":"2023-10-08","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Piastri","p3":"Hamilton"},
        {"round":18,"race":"US GP","circuit":"Circuit of the Americas","date":"2023-10-22","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Leclerc"},
        {"round":19,"race":"Mexico City GP","circuit":"Autodromo Hermanos Rodriguez","date":"2023-10-29","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Hamilton","p3":"Leclerc"},
        {"round":20,"race":"São Paulo GP","circuit":"Autodromo Jose Carlos Pace","date":"2023-11-05","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Hamilton"},
        {"round":21,"race":"Las Vegas GP","circuit":"Las Vegas Strip Circuit","date":"2023-11-18","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Sainz"},
        {"round":22,"race":"Abu Dhabi GP","circuit":"Yas Marina Circuit","date":"2023-11-26","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Hamilton"},
    ],
    2024: [
        {"round":1,"race":"Bahrain GP","circuit":"Bahrain International Circuit","date":"2024-03-02","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Sainz","p3":"Leclerc"},
        {"round":2,"race":"Saudi Arabian GP","circuit":"Jeddah Corniche Circuit","date":"2024-03-09","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Perez","p3":"Sainz"},
        {"round":3,"race":"Australian GP","circuit":"Albert Park Circuit","date":"2024-03-24","winner":"Carlos Sainz","winner_team":"Ferrari","event_type":"WIN","p2":"Leclerc","p3":"Norris"},
        {"round":4,"race":"Japanese GP","circuit":"Suzuka Circuit","date":"2024-04-07","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Perez","p3":"Sainz"},
        {"round":5,"race":"Chinese GP","circuit":"Shanghai International Circuit","date":"2024-04-21","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Perez"},
        {"round":6,"race":"Miami GP","circuit":"Miami International Autodrome","date":"2024-05-05","winner":"Lando Norris","winner_team":"McLaren","event_type":"WIN","p2":"Verstappen","p3":"Leclerc"},
        {"round":7,"race":"Emilia Romagna GP","circuit":"Imola","date":"2024-05-19","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Leclerc"},
        {"round":8,"race":"Monaco GP","circuit":"Circuit de Monaco","date":"2024-05-26","winner":"Charles Leclerc","winner_team":"Ferrari","event_type":"WIN","p2":"Piastri","p3":"Sainz"},
        {"round":9,"race":"Canadian GP","circuit":"Circuit Gilles Villeneuve","date":"2024-06-09","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Sainz"},
        {"round":10,"race":"Spanish GP","circuit":"Circuit de Barcelona-Catalunya","date":"2024-06-23","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Russell"},
        {"round":11,"race":"Austrian GP","circuit":"Red Bull Ring","date":"2024-06-30","winner":"George Russell","winner_team":"Mercedes","event_type":"WIN","p2":"Norris","p3":"Piastri"},
        {"round":12,"race":"British GP","circuit":"Silverstone Circuit","date":"2024-07-07","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Norris","p3":"Verstappen"},
        {"round":13,"race":"Hungarian GP","circuit":"Hungaroring","date":"2024-07-21","winner":"Oscar Piastri","winner_team":"McLaren","event_type":"WIN","p2":"Norris","p3":"Russell"},
        {"round":14,"race":"Belgian GP","circuit":"Spa-Francorchamps","date":"2024-07-28","winner":"Lewis Hamilton","winner_team":"Mercedes","event_type":"WIN","p2":"Piastri","p3":"Sainz"},
        {"round":15,"race":"Dutch GP","circuit":"Circuit Zandvoort","date":"2024-08-25","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Leclerc"},
        {"round":16,"race":"Italian GP","circuit":"Autodromo Nazionale Monza","date":"2024-09-01","winner":"Charles Leclerc","winner_team":"Ferrari","event_type":"WIN","p2":"Piastri","p3":"Norris"},
        {"round":17,"race":"Azerbaijan GP","circuit":"Baku City Circuit","date":"2024-09-15","winner":"Oscar Piastri","winner_team":"McLaren","event_type":"WIN","p2":"Leclerc","p3":"Sainz"},
        {"round":18,"race":"Singapore GP","circuit":"Marina Bay Street Circuit","date":"2024-09-22","winner":"Lando Norris","winner_team":"McLaren","event_type":"WIN","p2":"Piastri","p3":"Hamilton"},
        {"round":19,"race":"US GP","circuit":"Circuit of the Americas","date":"2024-10-20","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Leclerc"},
        {"round":20,"race":"Mexico City GP","circuit":"Autodromo Hermanos Rodriguez","date":"2024-10-27","winner":"Carlos Sainz","winner_team":"Ferrari","event_type":"WIN","p2":"Norris","p3":"Leclerc"},
        {"round":21,"race":"São Paulo GP","circuit":"Autodromo Jose Carlos Pace","date":"2024-11-03","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Russell"},
        {"round":22,"race":"Las Vegas GP","circuit":"Las Vegas Strip Circuit","date":"2024-11-23","winner":"Carlos Sainz","winner_team":"Ferrari","event_type":"WIN","p2":"Verstappen","p3":"Hamilton"},
        {"round":23,"race":"Qatar GP","circuit":"Lusail International Circuit","date":"2024-12-01","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Piastri","p3":"Norris"},
        {"round":24,"race":"Abu Dhabi GP","circuit":"Yas Marina Circuit","date":"2024-12-08","winner":"Lando Norris","winner_team":"McLaren","event_type":"WIN","p2":"Leclerc","p3":"Sainz"},
    ],
    2025: [
        {"round":1,"race":"Australian GP","circuit":"Albert Park Circuit","date":"2025-03-16","winner":"Lando Norris","winner_team":"McLaren","event_type":"WIN","p2":"Piastri","p3":"Leclerc"},
        {"round":2,"race":"Chinese GP","circuit":"Shanghai International Circuit","date":"2025-03-23","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Piastri"},
        {"round":3,"race":"Japanese GP","circuit":"Suzuka Circuit","date":"2025-04-06","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Leclerc"},
        {"round":4,"race":"Bahrain GP","circuit":"Bahrain International Circuit","date":"2025-04-13","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Piastri","p3":"Leclerc"},
        {"round":5,"race":"Saudi Arabian GP","circuit":"Jeddah Corniche Circuit","date":"2025-04-20","winner":"Oscar Piastri","winner_team":"McLaren","event_type":"WIN","p2":"Norris","p3":"Russell"},
        {"round":6,"race":"Miami GP","circuit":"Miami International Autodrome","date":"2025-05-04","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Piastri","p3":"Norris"},
        {"round":7,"race":"Emilia Romagna GP","circuit":"Autodromo Enzo e Dino Ferrari","date":"2025-05-18","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Leclerc","p3":"Piastri"},
        {"round":8,"race":"Monaco GP","circuit":"Circuit de Monaco","date":"2025-05-25","winner":"Charles Leclerc","winner_team":"Ferrari","event_type":"WIN","p2":"Norris","p3":"Verstappen"},
        {"round":9,"race":"Spanish GP","circuit":"Circuit de Barcelona-Catalunya","date":"2025-06-01","winner":"Max Verstappen","winner_team":"Red Bull Racing","event_type":"WIN","p2":"Norris","p3":"Piastri"},
    ],
}

FUTURE_RACES_2025 = [
    {"round":10,"race":"Canadian GP","circuit":"Circuit Gilles Villeneuve","date":"2025-06-15","country":"Canada"},
    {"round":11,"race":"Austrian GP","circuit":"Red Bull Ring","date":"2025-06-29","country":"Austria"},
    {"round":12,"race":"British GP","circuit":"Silverstone Circuit","date":"2025-07-06","country":"UK"},
    {"round":13,"race":"Belgian GP","circuit":"Spa-Francorchamps","date":"2025-07-27","country":"Belgium"},
    {"round":14,"race":"Hungarian GP","circuit":"Hungaroring","date":"2025-08-03","country":"Hungary"},
    {"round":15,"race":"Dutch GP","circuit":"Circuit Zandvoort","date":"2025-08-31","country":"Netherlands"},
    {"round":16,"race":"Italian GP","circuit":"Autodromo Nazionale Monza","date":"2025-09-07","country":"Italy"},
    {"round":17,"race":"Azerbaijan GP","circuit":"Baku City Circuit","date":"2025-09-21","country":"Azerbaijan"},
    {"round":18,"race":"Singapore GP","circuit":"Marina Bay Street Circuit","date":"2025-10-05","country":"Singapore"},
    {"round":19,"race":"US GP","circuit":"Circuit of the Americas","date":"2025-10-19","country":"USA"},
    {"round":20,"race":"Mexico City GP","circuit":"Autodromo Hermanos Rodriguez","date":"2025-10-26","country":"Mexico"},
    {"round":21,"race":"São Paulo GP","circuit":"Autodromo Jose Carlos Pace","date":"2025-11-09","country":"Brazil"},
    {"round":22,"race":"Las Vegas GP","circuit":"Las Vegas Strip Circuit","date":"2025-11-22","country":"USA"},
    {"round":23,"race":"Qatar GP","circuit":"Lusail International Circuit","date":"2025-11-30","country":"Qatar"},
    {"round":24,"race":"Abu Dhabi GP","circuit":"Yas Marina Circuit","date":"2025-12-07","country":"UAE"},
]

DRIVER_STANDINGS = {
    2021: [
        {"driver":"Max Verstappen","team":"Red Bull Racing","points":395,"wins":10},
        {"driver":"Lewis Hamilton","team":"Mercedes","points":387,"wins":8},
        {"driver":"Valtteri Bottas","team":"Mercedes","points":226,"wins":1},
        {"driver":"Sergio Perez","team":"Red Bull Racing","points":190,"wins":1},
        {"driver":"Carlos Sainz","team":"Ferrari","points":164,"wins":0},
        {"driver":"Charles Leclerc","team":"Ferrari","points":159,"wins":0},
        {"driver":"Lando Norris","team":"McLaren","points":160,"wins":0},
        {"driver":"Daniel Ricciardo","team":"McLaren","points":115,"wins":1},
    ],
    2022: [
        {"driver":"Max Verstappen","team":"Red Bull Racing","points":454,"wins":15},
        {"driver":"Charles Leclerc","team":"Ferrari","points":308,"wins":3},
        {"driver":"Sergio Perez","team":"Red Bull Racing","points":305,"wins":2},
        {"driver":"George Russell","team":"Mercedes","points":275,"wins":1},
        {"driver":"Carlos Sainz","team":"Ferrari","points":246,"wins":1},
        {"driver":"Lewis Hamilton","team":"Mercedes","points":240,"wins":0},
        {"driver":"Lando Norris","team":"McLaren","points":122,"wins":0},
        {"driver":"Esteban Ocon","team":"Alpine","points":92,"wins":0},
    ],
    2023: [
        {"driver":"Max Verstappen","team":"Red Bull Racing","points":575,"wins":19},
        {"driver":"Sergio Perez","team":"Red Bull Racing","points":285,"wins":2},
        {"driver":"Lewis Hamilton","team":"Mercedes","points":234,"wins":0},
        {"driver":"Fernando Alonso","team":"Aston Martin","points":206,"wins":0},
        {"driver":"Charles Leclerc","team":"Ferrari","points":206,"wins":0},
        {"driver":"Lando Norris","team":"McLaren","points":205,"wins":0},
        {"driver":"Carlos Sainz","team":"Ferrari","points":200,"wins":1},
        {"driver":"George Russell","team":"Mercedes","points":175,"wins":0},
    ],
    2024: [
        {"driver":"Max Verstappen","team":"Red Bull Racing","points":437,"wins":9},
        {"driver":"Lando Norris","team":"McLaren","points":374,"wins":4},
        {"driver":"Charles Leclerc","team":"Ferrari","points":356,"wins":3},
        {"driver":"Oscar Piastri","team":"McLaren","points":292,"wins":2},
        {"driver":"Carlos Sainz","team":"Ferrari","points":290,"wins":2},
        {"driver":"George Russell","team":"Mercedes","points":245,"wins":1},
        {"driver":"Lewis Hamilton","team":"Mercedes","points":223,"wins":2},
        {"driver":"Sergio Perez","team":"Red Bull Racing","points":152,"wins":0},
    ],
    2025: [
        {"driver":"Max Verstappen","team":"Red Bull Racing","points":161,"wins":6},
        {"driver":"Lando Norris","team":"McLaren","points":131,"wins":1},
        {"driver":"Oscar Piastri","team":"McLaren","points":127,"wins":2},
        {"driver":"Charles Leclerc","team":"Ferrari","points":78,"wins":1},
        {"driver":"George Russell","team":"Mercedes","points":67,"wins":0},
        {"driver":"Andrea Kimi Antonelli","team":"Mercedes","points":44,"wins":0},
        {"driver":"Carlos Sainz","team":"Williams","points":40,"wins":0},
        {"driver":"Fernando Alonso","team":"Aston Martin","points":36,"wins":0},
    ],
}

TEAM_COLORS = {
    "Red Bull Racing":"#3671C6","Ferrari":"#E8002D","Mercedes":"#27F4D2",
    "McLaren":"#FF8000","Aston Martin":"#229971","Alpine":"#0093CC",
    "Williams":"#64C4FF","AlphaTauri":"#5E8FAA","Alfa Romeo":"#C92D4B",
    "Haas":"#B6BABD","RB":"#6692FF",
}

STOCK_MAP = {
    "Ferrari (RACE)": "RACE",
    "Liberty Media / F1 Group (FWONA)": "FWONA",
    "Mercedes-Benz (MBG.DE)": "MBG.DE",
    "Pirelli (PIRC.MI)": "PIRC.MI",
    "Flutter Entertainment (FLUT)": "FLUT",
    "Puma (PUM.DE)": "PUM.DE",
}

TEAM_STOCK_LINK = {
    "Ferrari (RACE)": {
        "team": "Ferrari",
        "why": "Ferrari is both the F1 team AND the car manufacturer. When Ferrari wins, it directly boosts brand prestige, which analysts link to higher car sales demand and stock price. The Ferrari stock (NYSE: RACE) moves measurably around race weekends — studies show an average +1.8% in the 3 days after a win.",
    },
    "Liberty Media / F1 Group (FWONA)": {
        "team": None,
        "why": "Liberty Media OWNS Formula 1 itself. Their revenue comes from race hosting fees, TV rights, and sponsorship. More exciting seasons = more viewers = higher ad revenue = higher stock. When F1 viewership records are broken (like the Netflix Drive to Survive effect), FWONA reacts positively.",
    },
    "Mercedes-Benz (MBG.DE)": {
        "team": "Mercedes",
        "why": "Mercedes-Benz AG funds the Mercedes AMG Petronas F1 team. Winning seasons are used in marketing campaigns ('The most successful team in F1 history'). Stock analysts track the halo effect — championship years correlate with higher luxury car sales, especially in Asia.",
    },
    "Pirelli (PIRC.MI)": {
        "team": None,
        "why": "Pirelli is the EXCLUSIVE tyre supplier for all of F1. Every single race uses their tyres. Pirelli's F1 deal runs until 2027 and is a massive marketing contract — their logo appears in every broadcast. Dramatic tyre failures or blow-outs create negative news cycles for their stock.",
    },
    "Flutter Entertainment (FLUT)": {
        "team": None,
        "why": "Flutter owns Betfair, FanDuel and other betting platforms. F1 race weekends generate millions in betting volume. Surprise results (underdogs winning, dominant champions crashing out) spike betting activity and revenue. Flutter's quarterly earnings spike around high-viewership races.",
    },
    "Puma (PUM.DE)": {
        "team": None,
        "why": "Puma supplies team kit and merchandise for Ferrari, Red Bull, and Mercedes. Championship wins drive merchandise sales — a Verstappen or Leclerc championship means more jersey sales. Puma's F1 merchandise revenue is a small but visible line in their quarterly reports.",
    },
}

# ── 2026 season data ─────────────────────────────────────────────────────────
from utils.teams_sponsors import (
    RACES_2026, FUTURE_RACES_2026, DRIVER_STANDINGS_2026, F1_TEAMS_2026,
)
ALL_RACES[2026]        = RACES_2026
DRIVER_STANDINGS[2026] = DRIVER_STANDINGS_2026
FUTURE_RACES      = FUTURE_RACES_2026
FUTURE_RACES_2025 = FUTURE_RACES_2026
