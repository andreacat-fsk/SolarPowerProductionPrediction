!pip install -r requirements.txt

from datetime import datetime
import pytz

from pvlive_api import PVLive

pvlive = PVLive()

def download_pvlive_by_gsp(start, end, include_national=True, extra_fields=""):
    data = None
    pvl = PVLive()
    min_gsp_id = 0 if include_national else 1
    for gsp_id in pvl.gsp_ids:
        if gsp_id < min_gsp_id:
            continue
        data_ = pvl.between(start=start, end=end, entity_type="gsp", entity_id=gsp_id,
                            dataframe=True, extra_fields=extra_fields)
        if data is None:
            data = data_
        else:
            data = pd.concat((data, data_), ignore_index=True)
    return data

print(data)