import os
from dotenv import load_dotenv
import json
import requests
import pandas as pd
import progressbar

load_dotenv()
API_KEY = os.getenv('API_KEY')


def find_all_indices_of(value, list_to_search):
    results = list()
    for i, list_value in enumerate(list_to_search):
        if type(value) is list:
            if list_value in value:
                results.append(i)
        else:
            if list_value == value:
                results.append(i)
    return results


def multi_index(list_to_index, indices):
    return [element for i, element in enumerate(list_to_index) if i in indices]


climate_sector_map = {
    "000081": "Climate change",
    "000811": "Climate mitigation",
    "000812": "Climate adaptation"
}

au_country_map = {
    "DZ": "Algeria",
    "AO": "Angola",
    "BJ": "Benin",
    "BW": "Botswana",
    "BF": "Burkina Faso",
    "BI": "Burundi",
    "CM": "Cameroon",
    "CV": "Cape Verde",
    "CF": "Central African Republic",
    "TD": "Chad",
    "KM": "Comoros",
    "CD": "Democratic Republic of the Congo",
    "CG": "Republic of the Congo",
    "DJ": "Djibouti",
    "EG": "Egypt",
    "GQ": "Equatorial Guinea",
    "ER": "Eritrea",
    "SZ": "Eswatini",
    "ET": "Ethiopia",
    "GA": "Gabon",
    "GM": "Gambia",
    "GH": "Ghana",
    "GN": "Guinea",
    "GW": "Guinea-Bissau",
    "CI": "Ivory Coast",
    "KE": "Kenya",
    "LS": "Lesotho",
    "LR": "Liberia",
    "LY": "Libya",
    "MG": "Madagascar",
    "MW": "Malawi",
    "ML": "Mali",
    "MR": "Mauritania",
    "MU": "Mauritius",
    "MA": "Morocco",
    "MZ": "Mozambique",
    "NA": "Namibia",
    "NE": "Niger",
    "NG": "Nigeria",
    "RW": "Rwanda",
    "EH": "Sahrawi Arab Democratic Republic",
    "ST": "São Tomé and Príncipe",
    "SN": "Senegal",
    "SC": "Seychelles",
    "SL": "Sierra Leone",
    "SO": "Somalia",
    "ZA": "South Africa",
    "SS": "South Sudan",
    "SD": "Sudan",
    "TZ": "Tanzania",
    "TG": "Togo",
    "TN": "Tunisia",
    "UG": "Uganda",
    "ZM": "Zambia",
    "ZW": "Zimbabwe",
}


transaction_type_map = {
    '3': 'Disbursement',
    '4': 'Expenditure',
    '5': 'Interest Payment',
}


finance_type_map = {
    "410": "Aid loan excluding debt reorganisation",
    "411": "Investment-related loan to developing countries",
    "412": "Loan in a joint venture with the recipient",
    "413": "Loan to national private investor",
    "414": "Loan to national private exporter",
    "421": "Standard loan",
    "422": "Reimbursable grant",
    "423": "Bonds",
    "424": "Asset-backed securities",
    "425": "Other debt securities",
    "431": "Subordinated loan",
    "432": "Preferred equity",
    "433": "Other hybrid instruments",
    "451": "Non-banks guaranteed export credits",
    "452": "Non-banks non-guaranteed portions of guaranteed export credits",
    "453": "Bank export credits",
}


def main():
    # Use the IATI Datastore API to fetch all titles for a given publisher
    rows = 1000
    next_cursor_mark = '*'
    current_cursor_mark = ''
    results = []
    with progressbar.ProgressBar(max_value=1) as bar:
        while next_cursor_mark != current_cursor_mark:
            url = (
                'https://api.iatistandard.org/datastore/transaction/select'
                '?q=(reporting_org_ref:"44000" AND '
                'transaction_transaction_type_code:("3" OR "4" OR "5") AND '
                'transaction_finance_type_code:4* AND '
                'sector_code:("000081" OR "000811" OR "000812"))'
                '&sort=id asc'
                '&wt=json&fl=iati_identifier,title_narrative,'
                'sector_code,sector_percentage,sector_vocabulary,'
                'recipient_country_code,recipient_country_percentage,'
                'transaction_value,transaction_transaction_date_iso_date,transaction_transaction_type_code,'
                'transaction_finance_type_code'
                '&rows={}&cursorMark={}'
            ).format(rows, next_cursor_mark)
            api_json_str = requests.get(url, headers={'Ocp-Apim-Subscription-Key': API_KEY}).content
            api_content = json.loads(api_json_str)
            if bar.max_value == 1:
                bar.max_value = api_content['response']['numFound']
            transactions = api_content['response']['docs']
            len_results = len(transactions)
            current_cursor_mark = next_cursor_mark
            next_cursor_mark = api_content['nextCursorMark']
            for transaction_number, transaction in enumerate(transactions):
                transaction_type_code = transaction['transaction_transaction_type_code'][0]
                transaction_finance_type_code = transaction['transaction_finance_type_code'][0]
                if not transaction_finance_type_code.startswith('4'):
                    continue
                transaction_dict = dict()
                transaction_dict['iati_identifier'] = transaction['iati_identifier']
                transaction_dict['transaction_number'] = transaction_number
                transaction_dict['title'] = transaction['title_narrative'][0]
                transaction_dict['date'] = transaction['transaction_transaction_date_iso_date'][0]
                transaction_dict['transaction_type'] = transaction_type_map[transaction_type_code]
                transaction_dict['finance_type'] = finance_type_map[transaction_finance_type_code]
                transaction_value = float(transaction['transaction_value'][0])
                transaction_dict['original_transaction_value'] = transaction_value
                reporting_org_v2_indices = find_all_indices_of('98', transaction['sector_vocabulary'])
                reporting_org_sector_codes = multi_index(transaction['sector_code'], reporting_org_v2_indices)
                reporting_org_sector_percentages = multi_index(transaction['sector_percentage'], reporting_org_v2_indices)
                sector_relevant = False
                for climate_sector in climate_sector_map.keys():
                    if climate_sector in reporting_org_sector_codes:
                        sector_relevant = True
                        break
                recipient_relevant = False
                for recipient in au_country_map.keys():
                    if 'recipient_country_code' in transaction.keys() and recipient in transaction['recipient_country_code']:
                        recipient_relevant = True
                        break
                if not (sector_relevant and recipient_relevant):
                    continue
                climate_sector_indices = find_all_indices_of(list(climate_sector_map.keys()), reporting_org_sector_codes)
                au_recipient_indices = find_all_indices_of(list(au_country_map.keys()), transaction['recipient_country_code'])
                for climate_sector_index in climate_sector_indices:
                    for au_recipient_index in au_recipient_indices:
                        split_dict = transaction_dict.copy()
                        climate_sector_code = reporting_org_sector_codes[climate_sector_index]
                        split_dict['climate_sector_name'] = climate_sector_map[climate_sector_code]
                        climate_sector_percentage = float(reporting_org_sector_percentages[climate_sector_index]) / 100
                        split_dict['climate_sector_percentage'] = reporting_org_sector_percentages[climate_sector_index]
                        au_recipient_code = transaction['recipient_country_code'][au_recipient_index]
                        split_dict['recipient_name'] = au_country_map[au_recipient_code]
                        au_recipient_percentage = float(transaction['recipient_country_percentage'][au_recipient_index]) / 100
                        split_dict['recipient_percentage'] = transaction['recipient_country_percentage'][au_recipient_index]
                        split_dict['split_transaction_value'] = transaction_value * climate_sector_percentage * au_recipient_percentage
                        results.append(split_dict)
            if bar.value + len_results <= bar.max_value:
                bar.update(bar.value + len_results)
    
    # Collate into Pandas dataframe
    df = pd.DataFrame.from_records(results)

    # Write to disk
    df.to_csv(
        os.path.join('data', 'world_bank_climate_loans_au.csv'),
        index=False,
    )


if __name__ == '__main__':
    main()