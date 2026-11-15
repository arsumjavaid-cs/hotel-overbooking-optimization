"""Build 2: reproducible, ILLUSTRATIVE hotel overbooking retrospective.
Usage: python build2_analysis.py --csv ../hotels.csv --out results --city-capacity 150 --resort-capacity 100
Not deployable: actual capacities, booking snapshots, booking IDs and cancellation times absent.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

CAT = ['hotel', 'arrival_date_month', 'market_segment', 'distribution_channel', 'deposit_type',
       'customer_type', 'reserved_room_type', 'meal']
NUM = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults',
       'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
       'previous_bookings_not_canceled', 'adr']
LEAKY = ['reservation_status', 'reservation_status_date', 'assigned_room_type',
         'booking_changes', 'days_in_waiting_list']


def load(csv):
    df = pd.read_csv(csv)
    df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'].astype(str) + '-' + df['arrival_date_day_of_month'].astype(str),
        format='%Y-%B-%d', errors='raise')
    df['shows'] = 1 - df['is_canceled']
    return df


def evaluate(frame, capacity, k_col):
    tmp=frame.copy()
    k=tmp[k_col].clip(lower=0).astype(int)
    # ASSUMPTION: incremental k reservations each show up and stay one arrival night.
    tmp['empty_rooms'] = np.maximum(capacity-tmp['actual_shows']-k, 0)
    tmp['walks'] = np.maximum(tmp['actual_shows']+k-capacity,0)
    tmp['euros'] = tmp['empty_rooms']*tmp['rate_eur'] + 300*tmp['walks']
    walks=int(tmp['walks'].sum())
    arriving=int((tmp['actual_shows']+k).sum())
    return {'cost_eur':round(float(tmp['euros'].sum()),2), 'walks':walks,
            'arrivals':arriving, 'walks_per_1000':round(1000*walks/max(arriving,1),3),
            'empty_rooms':int(tmp['empty_rooms'].sum()), 'extra_rooms':int(k.sum()),
            'decision_nights':int((k>0).sum())}


def make_daily(df, predictions, capacities):
    a=df[['hotel','arrival_date','shows','adr']].copy()
    a['predicted_shows']=predictions
    # Negative ADR invalid for a cost unit; a missing/zero rate is handled via positive hotel training median.
    a['clean_adr']=a['adr'].where(a['adr']>0,np.nan)
    daily=a.groupby(['hotel','arrival_date'],as_index=False).agg(
        bookings=('shows','size'),actual_shows=('shows','sum'),predicted_shows=('predicted_shows','sum'),
        rate_eur=('clean_adr','median'))
    for hotel, cap in capacities.items():
        mask=daily.hotel.eq(hotel)
        daily.loc[mask,'capacity_assumed']=cap
    daily['rate_eur']=daily['rate_eur'].fillna(0)
    return daily


def policy_k(daily, method, param, historical_cancel):
    capacity=daily['capacity_assumed'].to_numpy()
    booked=daily['bookings'].to_numpy()
    eligible=(booked>=capacity)  # No recommendation if bookings did not reach assumed capacity.
    if method=='zero':
        k=np.zeros(len(daily))
    elif method=='flat':
        k=np.full(len(daily), int(param))
    else:
        if method=='historical_rate':
            p=np.array([historical_cancel[h] for h in daily['hotel']])
            predicted=booked*(1-p)
        elif method=='logistic':
            predicted=daily['predicted_shows'].to_numpy()
        else: raise ValueError(method)
        # param = safety buffer; larger buffer => less overbooking
        k=np.rint(capacity-predicted-float(param))
    k=np.clip(k,0,20).astype(int)
    return np.where(eligible,k,0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='../hotels.csv')
    ap.add_argument('--out',default='results')
    ap.add_argument('--city-capacity',type=int,default=150)
    ap.add_argument('--resort-capacity',type=int,default=100)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True, exist_ok=True)
    caps={'City Hotel':args.city_capacity,'Resort Hotel':args.resort_capacity}
    df=load(args.csv)
    train=df[df.arrival_date<'2016-07-01'].copy()
    val=df[(df.arrival_date>='2016-07-01')&(df.arrival_date<'2017-01-01')].copy()
    test=df[df.arrival_date>='2017-01-01'].copy()
    if not len(train) or not len(val) or not len(test): raise ValueError('Missing time split')
    medians=train.loc[train.adr>0].groupby('hotel').adr.median().to_dict()
    historical_cancel=train.groupby('hotel').is_canceled.mean().to_dict()
    model=Pipeline([
      ('prep',ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),
                         ('scale',StandardScaler())]), NUM),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                         ('onehot',OneHotEncoder(handle_unknown='ignore'))]), CAT)])),
      ('classifier', LogisticRegression(max_iter=400,solver='liblinear',random_state=42))])
    model.fit(train[NUM+CAT],train['shows'])
    valp=model.predict_proba(val[NUM+CAT])[:,1]
    testp=model.predict_proba(test[NUM+CAT])[:,1]
    # Prediction metrics are diagnostics only, NOT the business objective.
    diag={'validation_auc':round(float(roc_auc_score(val.shows,valp)),4),
          'validation_brier':round(float(brier_score_loss(val.shows,valp)),4),
          'test_auc':round(float(roc_auc_score(test.shows,testp)),4),
          'test_brier':round(float(brier_score_loss(test.shows,testp)),4)}
    vdaily=make_daily(val,valp,caps)
    tdaily=make_daily(test,testp,caps)
    for daily in (vdaily,tdaily):
        for h in caps:
            sel=daily.hotel.eq(h)
            daily.loc[sel,'rate_eur']=daily.loc[sel,'rate_eur'].replace(0,np.nan).fillna(medians[h])
    candidate=[]
    methods={'zero':[0], 'flat':list(range(0,21)),
             'historical_rate':[0,2,5,8,10,15,20,25,30,40,50],
             'logistic':[0,2,5,8,10,15,20,25,30,40,50]}
    for hotel, capacity in caps.items():
        v=vdaily[vdaily.hotel.eq(hotel)].copy()
        for method,params in methods.items():
            for param in params:
                v['k']=policy_k(v,method,param,historical_cancel)
                m=evaluate(v,capacity,'k')
                candidate.append({'hotel':hotel,'method':method,'parameter':param,
                                  'eligible':m['walks_per_1000']<=10,**m})
    cv=pd.DataFrame(candidate)
    cv.to_csv(out/'validation_grid.csv',index=False)
    # Freeze one parameter per method per hotel using validation only.
    selected=[]
    for (hotel,method),grp in cv.groupby(['hotel','method'],sort=True):
        valid=grp[grp.eligible]
        pool=valid if len(valid) else grp
        best=pool.sort_values(['cost_eur','walks_per_1000','parameter']).iloc[0]
        selected.append({'hotel':hotel,'method':method,'parameter':int(best.parameter),
                         'validation_feasible':bool(best.eligible),
                         'validation_cost_eur':float(best.cost_eur),
                         'validation_walks_per_1000':float(best.walks_per_1000)})
    sel=pd.DataFrame(selected)
    sel.to_csv(out/'frozen_policies.csv',index=False)
    # Final test: policies and parameters never change based on 2017 outcomes.
    rows=[]; daily_examples=[]
    for r in sel.itertuples(index=False):
        t=tdaily[tdaily.hotel.eq(r.hotel)].copy()
        t['k']=policy_k(t,r.method,r.parameter,historical_cancel)
        m=evaluate(t,caps[r.hotel],'k')
        rows.append({'hotel':r.hotel,'method':r.method,'parameter_frozen':r.parameter,
                     'validation_feasible':r.validation_feasible,
                     'test_under_ceiling':m['walks_per_1000']<=10,**m})
        t['method']=r.method
        daily_examples.append(t)
    res=pd.DataFrame(rows)
    baseline=res[res.method.eq('zero')].set_index('hotel')['cost_eur'].to_dict()
    res['savings_vs_zero_eur']=res.apply(lambda r:round(baseline[r.hotel]-r.cost_eur,2),axis=1)
    res.to_csv(out/'test_comparison.csv',index=False)
    pd.concat(daily_examples,ignore_index=True).to_csv(out/'test_daily_decisions.csv',index=False)
    counts=df.groupby(['arrival_date_year','hotel']).agg(bookings=('shows','size'),cancellation_rate=('is_canceled','mean')).reset_index()
    counts.to_csv(out/'dataset_profile.csv',index=False)
    audit={'rows':len(df),'columns_original':32,'identical_rows_do_not_auto_delete':int(df.drop(columns=['arrival_date','shows']).duplicated().sum()),
           'missing':df.isna().sum().astype(int).to_dict(),
           'split_records':{'train':len(train),'validation':len(val),'test':len(test)},
           'model_metrics':diag,'assumed_capacity':caps,
           'notes':['This is a retrospective arrival-night simulation, NOT a live hotel recommendation.',
                    'Unknown actual room capacities; provided values are hypothetical scenarios.',
                    'No booking snapshots, cancellation times, or booking IDs.',
                    'Extra reservations are assumed guaranteed arrivals; this is a modeling simplification.',
                    'Same-day arrival cohorts are used instead of complete multiday occupied room-nights.',
                    '2017 results are a single frozen-policy evaluation; do not tune on test.',
                    'Zero/negative ADR replaced with positive training hotel median where daily median unavailable.',
                    'Walk ceiling uses modeled arriving guests as denominator; clarify exact client definition.',
                    'Booking-time availability of each feature requires operational confirmation.']}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('DATA:',len(df),'Train:',len(train),'Val:',len(val),'Test:',len(test))
    print('MODEL:',diag)
    print('FROZEN POLICIES:\n',sel.to_string(index=False))
    print('2017 TEST:\n',res.to_string(index=False))
    print('Files in',out.resolve())

if __name__=='__main__': main()
