Which columns did the hotel know on the day of booking?
Owner: Data Lead. Closes: part of ladder stage 1. Used by: src/features.py, the PRD and slide 4.
Put every column in exactly one bin, by when the hotel learns its value:
1.	Fixed at booking: known on the day the booking is made, and it does not change.
2.	Amendable: known at booking but it can change before arrival, or it is only partly known at booking.
3.	After the fact: written at check-in, during the stay, or when the outcome is already known.
Only the first bin feeds the MVP. A second-bin column may be tried with its caveat written beside it. A third-bin column never feeds a method. The dataset article (Antonio, de Almeida and Nunes, 2019) says how several columns were recorded; cite it where it settles a bin.
The evidence column holds the number or the source that settled the bin, for example a rate you computed, or a sentence from the dataset article.
Column	Bin (1, 2 or 3)	Why	Evidence
Column	Your bin	My bin	Why	Evidence
hotel	1	1	Resort or city property, fixed at booking.	Article: dataset covers two hotels (resort H1, city H2).
is_canceled	3	3	This is the target; the outcome isn’t known at booking.	Compute: cross-tab with reservation_status.
lead_time	1	1*	Days from PMS entry to arrival, fixed once the booking exists.	Article defines it as entry date to arrival date. *Caveat: it shifts if the arrival date is amended. Compute: AUC on booking_changes == 0 rows only.
arrival_date_year / month / week_number / day_of_month	1	1*	Dates are the planning anchor. Week number is redundant with the date.	*Dataset stores the final date, not the original. Same robustness check as above.
stays_in_weekend_nights / stays_in_week_nights	1	2	Length of stay can be amended.	Article’s booking_changes definition covers amendments from entry until check-in or cancellation.
adults / children / babies	2	2	Party size can change before arrival.	Same booking_changes definition. Compute: share of bookings with booking_changes > 0.
meal	2	2	Board basis can be amended.	Same as above.
country	1	1*	Guest origin, normally entered at booking.	*Compute: missing rate (~0.4% from memory). Some values may be completed at check-in.
market_segment / distribution_channel	1	1	Set by the booking source.	Compute: share of “Undefined” values.
is_repeated_guest	1	1*	Guest history flag.	*Caveat: may be set at check-in. Compute: consistency with previous_bookings_not_canceled > 0.
previous_cancellations / previous_bookings_not_canceled	1	1	History counted before this booking.	Article: counts of the guest’s previous bookings.
reserved_room_type	1	1*	Room code requested at booking.	*Upgrades go into assigned_room_type, but amendments could change it. Compute: reserved vs assigned mismatch rate.
assigned_room_type	2/3	3	Assigned at or near check-in, and it differs from the reserved type for operational reasons, including overbooking. It’s partly a result of the policy you are modelling.	Article’s attribute description says it differs for hotel operation reasons such as overbooking. Compute: mismatch rate by is_canceled.
booking_changes	2	2 (caveat)	The count runs until check-in or cancellation, so it is truncated by the outcome. Cancelled bookings have less time to accumulate changes.	Article definition (entry to check-in or cancellation). Compute: cancel rate for 0 vs >0 changes.
deposit_type	1	1, leakage-test first	Known at booking, but “Non Refund” is reported to have a ~99% cancel rate, which is implausible as a real behaviour.	Compute: cancel rate by deposit type. If it’s ~99%, the field was likely set or updated after the outcome, so treat it as bin 3.
agent	1	1	Who made the booking.	Article: ID of the travel agency. NULL means not applicable, not missing.
company	1	1*	Entity responsible for payment.	*~94% null. NULL means not applicable. May be filled later.
days_in_waiting_list	2	2	Final only once the booking is confirmed to the guest.	Article: days on the waiting list before confirmation. Compute: share of zeros (very high).
customer_type	1	1*	Contract, Group, Transient or Transient-party.	*Transient-party depends on linked bookings, which may be identified later.
adr	2	2 (caveat)	A rate at booking, but it can change with amendments.	Article defines it as lodging transactions divided by nights, so it is a derived value. Compute: share of zero ADR, and ADR vs deposit_type.
required_car_parking_spaces	2	3?	Often recorded close to or at arrival.	Compute: I recall near-zero values for cancelled bookings. If true, it’s outcome-driven and goes to 3.
total_of_special_requests	2	2	Requests accumulate before arrival.	Compute: mean by is_canceled. A large gap means it is partly outcome-driven.
reservation_status	2/3	3	It is the outcome (Canceled, Check-Out, No-Show).	Article: last status of the reservation. Compute: is_canceled == 1 iff status is Canceled or No-Show.
reservation_status_date	2/3	3	Date the final status was set, so it is the cancellation or checkout date.	Article: used with reservation_status to find when it was cancelled or checked out. Compute: it gives days-to-arrival at cancellation, which is useful for evaluation but never as a feature.
