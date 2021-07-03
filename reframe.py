from get_value import values,scaled_macd,scaled
from util import series_to_supervised


# frame as supervised learning
reframed_open = series_to_supervised(scaled, 1, 1)
reframed_close = series_to_supervised(scaled, 1, 1)
reframed_high = series_to_supervised(scaled, 1, 1)
reframed_low = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
# reframed.drop(reframed.columns[[5,6,7,8,9,11]], axis=1, inplace=True)

reframed_close.drop(reframed_close.columns[[5,6,7]], axis=1, inplace=True)
reframed_high.drop(reframed_high.columns[[4,6,7]], axis=1, inplace=True)
reframed_low.drop(reframed_low.columns[[4,5,7]], axis=1, inplace=True)
reframed_open.drop(reframed_open.columns[[4,5,6]], axis=1, inplace=True)