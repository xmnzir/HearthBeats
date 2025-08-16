# ============================================================
# File: Data_Gen.py
# Author: Mohammed Munazir
# Description: Data Generation Script for Synthetic Data
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

num_rows = 1500


content_samples = [
    "If I need any repairs  done they come straight out or come out the next day",
    "Keep you updated",
    "I have had no issues.",
    "Very good housing and very friendly service",
    "great service",
    "Made sure that I got my heating and hot water back on the same day he arrived",
    "Generally rent is fair and necessary work and repairs are carried out saisfactorily",
    "Servicing is kept up to date and I’ve just had a full electrical check which makes me feel safe",
    "so far jobs have been carried out as promised",
    "very prompt with repairs. The staff are great, always ready to help.  inside doors are very dated, but all in all everything else is good.",
    "From getting the phone call to offer me the house nicola was fantastic throughout it all",
    "because you are honnest",
    "Efficiency, reliability, prompt responses.",
    "Helpfull",
    "Very good service",
    "Workers great help respectful",
    "Prompt repair, lovely repair man and a job well done",
    "All and any issues resolved quickly",
    "Every time I have reported anything I have needed doing the people have come out and have been responsible and courteous. They have also been clean and tidy and left the house as they found it and they have also done the job perfectly. Other than repairs, everything is also OK.",
    "When I want anything doing it is done in good time and the workmen are always polite. I have lived here for a few years so I know them pretty well, apart from the repairs everything is fine with them.",
    "If I am needing repairs doing they will do them! There is not much that they will not do. I have lived here for 15 months now so I am still fairly new it has all been brilliant including the process for moving in to the property.",
    "As soon as you call and report something they come and get the job done, there is no waiting about. Apart from the repairs everything is also OK, the neighborhood is nice and quiet, we are in a cul-de-sac.",
    "Any repairs or things that go wrong, North Star do them as soon as they can do.",
    "I have had no problems at all with them. North Star do what is needed at the time, the repairs.",
    "It is just a good and quick service from them and they are friendly. I have lived here for five years now so know them pretty well and feel safe and sound in my new build home.",
    "If we got a problem we ring them up, they know I am old and disabled and they are very quick to respond.",
    "I am more than satisfied, they are there if you want anything and everything always gets dealt with. It does not matter how small the issue is, they will deal with it. It was strange when we were in lockdown but they still came out and what Ben needed was sorted by them until I arrived.",
    "When you report jobs they come and do them straight away. I have lived here for ten years so I know them pretty well, apart from the repairs it is all OK, I feel safe and secure in my home.",
    "I had a problem with my boiler that was going on for a few months. I contacted the housing officer directly and it was sorted within a week.",
    "I just think that they are great. Of all the other Landlords I have had they have never done repairs but these are different. I moved in in August, I was gifted carpets and had a paint card to decorate. I have had the boiler down twice, I cannot fault them, it is working OK at the moment. We had knocked the radiator and that caused the problem.",
    "Everything is alright. The housing is brilliant. Everyone at North Star is brilliant. Brenda at North Star is absolutely brilliant. I have no faults with them.",
    "They are alright, I have no problems with them at all. I have lived here for over a year, it was a smooth process for me moving in and I have settled well. The move was a good one and I have had no issues.",
    "Everything is going good, when I need something it is done quickly.",
    "The maintenance men, I had a gutter leak and they came and mended it in one place but they did not check the rest of it. So, next time it rained leaked elsewhere. They did not check it properly. There is nothing really that they do that is that good.",
    "I have never had any problems at all with them. We have a brilliant Housing Officer and they do a lot of extras with regard to energy efficiency and the cost of living crisis. From the information they have sent they have been and checked all properties for extra efficiency on heating and made improvements where necessary and this has helped in general. Jane is the Housing Officer and she is great, if there is anything she can do she will. Over the years she has helped with my Daughter and arranged improvement work to be carried out. She is always available and will put in place anything that is needed.",
    "I say that because they give a good service.",
    "If we have ever reported any defects, they done them immediately.",
    "They do a good job.",
    "They do come whenever I need them and solve any problems. At the moment the only problem is when something is getting repaired I have not been getting informed that they are coming. They say they have texted me and have tried to call but they just arrive without any notification which is difficult as I am a working single mother.",
    "When they come out they are great and do a good job. I do have a complaint as the girl who is in charge around here was up here. My husband told her that the front door archway the concrete is coming off and nothing was done about it.",
    "Every time I have to call them out they do a good job. I have lived here for 15 years so I know them pretty well. Apart from the repairs everything is good, the Neighbours and the area are fine.",
    "I am happy",
    "I am just happy, they come and see me, they come and check up on me regularly. The repair system is good though I am waiting for a repair at the moment. I have lived here for about twenty years so I know them pretty well.",
    "Every time I ring them for a repair I get it done straight away, I do not have to wait months for jobs to be done. Apart from the repairs everything is OK, the neighborhood is OK and I feel safe and secure in my home!",
    "They help me out a lot.",
    "I have no problem with them.",
    "North Star do what they can. When you report a repair, they try and come out as quick as they can.",
    "I got everything sorted straight away, I had an emergency and they came out within an hour.",
    "It is a brand new bungalow and everything in it is modern. I moved in in November 2021, so about 18 months ago. The process of moving in was all fine and there have been no major problems since though the garden is a bit large for me to maintain.",
    "I love it here because they are very nice and friendly. I also get my repairs done quickly and it is better to where I used to live.",
    "Whenever I have had a problem and rung up the office they have sorted it within a day. Nothing has been a problem for them and they have resolved my problems quickly and efficiently.",
    "This is because their customer service is excellent and any problems are resolved well. I have had no problems with North Star whatsoever.",
    "This is because I have had no problems and there is nothing to complain about.",
    "It is a good service."
]


customer_ids = [
    "BREN010210010", "Anonymous", "WORT020150001", "MILT010770106", "PLAN010590001",
    "Anonymous", "ST P030840009", "BALD010050001", "Anonymous", "PARK030100001",
    "PION020430001", "Anonymous", "Anonymous", "NENT010050006", "LARV020050005",
    "AIDA010040003", "AYRE030310001", "MITC021150001", "COPE010190002", "WELL020780003",
    "MANS010810009", "WINS010010001", "EMBL010020002"
]


management_areas = [
    "Hartlepool", "Thornaby", "Stockton", "Sunderland", "Darlington", "Teesdale", "Middlesbrough",
    "Skinningrove", "Evenwood", "Staindrop", "Consett", "Cockfield", "Stokesley", "Stanley",
    "NORTON", "THIRSK", "Barnard Castle", "REDCAR", "Carlinhow", "Cotherstone", "Gainford", "Etherley"
]


start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 1, 1)
dates = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(num_rows)]


data = {
    "publishedx_at": dates,
    "brand": np.random.choice(["Brand A", "Brand B", "Brand C", "Brand D"], num_rows),
    "contentx": np.random.choice(content_samples, num_rows),
    "Customer Idx": np.random.choice(customer_ids, num_rows),
    "Question": np.random.choice([
        "How can I improve model precision?",
        "What causes high variance in predictions?",
        "How do I handle missing values?",
        "What hyperparameters should I tune?",
        "Why is inference so slow on new data?"
    ], num_rows),
    "Tenure Typex": np.random.choice(
        ["Leasehold", "Freehold", "Shared Ownership", "Private Rental"], num_rows
    ),
    "Management areax": np.random.choice(management_areas, num_rows),
    "Age Groupx": np.random.choice(
        ["18-25", "26-35", "36-45", "46-60", "60+"], num_rows
    ),
    "Ethnicityx": np.random.choice(
        ["White", "Black", "Asian", "Mixed", "Other"], num_rows
    ),
    "Property Typex": np.random.choice(
        ["Flat", "House", "Bungalow", "Maisonette", "Studio"], num_rows
    ),
}

df_synthetic = pd.DataFrame(data)


print(df_synthetic.head())


df_synthetic.to_csv('synthetic_data.csv', index=False)
