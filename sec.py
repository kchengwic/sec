import re, urllib2, itertools, datetime, csv, os, string, sys, __builtin__
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import xml.etree.ElementTree
import logging.config
from requests import get
import multiprocessing as mp
import Levenshtein
from cleanco import cleanco
import time

SEC_PREFIX = 'https://www.sec.gov'
SEC_SIC = 'https://www.sec.gov/info/edgar/siccodes.htm'
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def scrape_cik_by_ticker_re(length):
    """
    Another implementation for scraping the CIK by different combo of ticker of length L. It then writes the found
    ticker-cik dict to disk.
    :param L: int
    :return: None
    """
    processes = 4
    ticker_list = list(itertools.product(ALPHABET, repeat=length))
    ticker_list = np.array([''.join(subset) for subset in ticker_list])
    size = len(ticker_list) / processes
    # divide the ticker list into sub-lists for multiprocessing
    c = [ticker_list[i:i + size] for i in xrange(0, len(ticker_list), size)]
    pool = mp.Pool(processes=processes)
    result = pool.map(get_cik_by_re, c)
    d = {}
    for res_dict in result:
        d = merge_two_dicts(d, res_dict)
    write_to_disk('./', 'ticker_cik_len' + str(length) + '_another.csv', d)

def get_cik_by_re(ticker_list):
    """
    Called by scrape_cik_by_ticker for multiprocessing to speed up the scrape process.
    :param ticker_list: List of string
    :return: dict (key: ticker, value: cik)
    """
    URL = 'http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany'
    CIK_RE = re.compile(r'.*CIK=(\d{10}).*')
    cik_dict = {}
    for ticker in ticker_list:
        results = CIK_RE.findall(get(URL.format(ticker)).content)
        if len(results):
            cik_dict[ticker] = str(results[0])
            # print 'found: ' + ticker + ' ' + str(results[0])
    return cik_dict


def scrape_cik_by_ticker_xml(length=6):
    """
    One implementation of generate all possible combo of ticker with 1 letter up to length letters.
    :param length: int
    :return: None
    """
    start = datetime.datetime.now()
    for L in range(1, length+1):  # scrape combo of 1 letter up to length letters. However, len = 6 will most likely overflow the memory
        # for prefix in '':
        ticker_list = list(itertools.product(ALPHABET, repeat=L))
        ticker_list = np.array([''.join(subset) for subset in ticker_list])#prefix + ''.join(subset) for subset in ticker_list])
        vfunc = np.vectorize(get_cik_from_xml)
        vcik = vfunc(ticker_list)
        np.savetxt('./ticker_cik_len_' + str(L) + '.csv', np.column_stack((ticker_list, vcik)),#prefix + '.csv', np.column_stack((ticker_list, vcik)),
                   delimiter=',', newline='\n', fmt='%s')
        end = datetime.datetime.now()
        print 'Time used for %s letter combo: %s' % (str(L), convert_timedelta_to_hour_min_sec(end - start))
        start = end

def get_cik_from_xml(ticker):
    cik = None
    try:
        xml_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s' \
                  '&type=8-K&dateb=&owner=&start=0&count=100&output=xml' % ticker
        html = get_html(xml_url)
        e = xml.etree.ElementTree.parse(html)
        root = e.getroot()
        cik = root.findall('companyInfo')[0].find('CIK').text
    except xml.etree.ElementTree.ParseError as e:
        pass
    return cik


def scrape_sic(url=SEC_SIC):
    """
    Scrape the SIC code URL and return the table that contains SIC code, office code and Industry title.
    :param url: string
    :return: DataFrame
    """
    table = pd.read_html(url)[3]
    table = table.dropna(axis=[0, 1], how='all')
    table.columns = table.iloc[0]
    table = table[1:]
    table = table.reset_index(drop=True)
    return table

def scrape_cik_by_sic(sic):
    """
    Scrape all CIK available under a SIC and return a dataframe that contains CIK, Company name and its
    State/Country code. None will be returned if no companies are found or the SIC is invalid.
    :param sic: string
    :return: DataFrame
    """
    df_list = []
    start = 0
    while True:
        try:
            url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC=%s&owner=&start=%s&count=100' % (sic, start)
            df_list += pd.read_html(url, header=0)
            start += 100
            if len(df_list[-1].columns) != 3:
                logger.error('The dataframe for SIC: %s has problem' % sic)
                return None
        except ValueError as e:
            if str(e) == 'No tables found':
                if df_list:
                    break
                return None
    df = pd.concat(df_list).reset_index(drop=True)
    return df

def get_all_sic_cik_from_sec(path):
    """
    Scrape all CIK from all SIC on SEC's page.
    :param path: string. Path for the output file
    :return: None
    """
    try:
        header = True
        sic_df = scrape_sic()
        for index, row in sic_df.iterrows():
            sic = row[0]
            industry_title = row[2]
            cik_df = scrape_cik_by_sic(sic)
            if cik_df is not None:
                cik_df['SIC'] = sic
                cik_df['Industry Title'] = industry_title
                cik_df.to_csv(path, mode='ab', index=False, header=header, encoding='utf-8')
                if os.path.exists(path):
                    header = False
    except Exception as e:
        logger.error(str(e))



def get_result_from_xml(ticker, attr, start, form):
    """
    Parse the XML and return the attribute body. In below URL, the attr can be either companyInfo or results under
    the root companyFilings.
    :param ticker: String
    :param attr: String
    :param start: int
    :param form: String, default = 8-K
    :return: xml.etree.ElementTree object
    """
    results = html = None
    try:
        xml_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=%s' \
                  '&type=%s&dateb=&owner=&start=%s&count=100&output=xml' % (ticker, form, str(start))
        html = get_html(xml_url)
        e = xml.etree.ElementTree.parse(html)
        root = e.getroot()
        results = root.findall(attr)
    except xml.etree.ElementTree.ParseError as e:
        soup = BeautifulSoup(html, 'lxml')
        if re.search('No matching Ticker Symbol.', soup.find('h1').text, re.IGNORECASE):
            logger.warn("No matching Ticker Symbol: %s" % ticker)
        else:
            logger.warn("Some other xml.etree.ElementTree.ParseError")
    except Exception as e:
        logger.warn("Some other Exception: %s" % e.message)
    return results

def get_html(url, sleep_time=0.5):
    html = urllib2.urlopen(url)
    # sleep for sleep_time sec to ensure it does not excess the 10 requests per second limit in SEC
    time.sleep(sleep_time)
    return html

def retrieve_doc_by_type(index_url, type='EX-2.1'):
    """
    Read the index url and search for EX-2.1 document URL.
    :param index_url: String
    :param type: String
    :return: a List of the EX-2.1 document URL
    """
    docs = []
    html = get_html(index_url).read()
    soup = BeautifulSoup(html, 'lxml')
    tabulka = soup.find("table", {"class": "tableFile"})

    for row in tabulka.findAll('tr')[1:]:
        # exclude the first row which is the table header, i.e. Seq, Description, Document, Type, Size
        col = row.findAll('td')
        form_type = col[3].renderContents().strip()
        # if type == form_type:
        if re.search(type, form_type, re.IGNORECASE):
            # logger.info( __builtin__.type(col[2].find_all('a', href=True)[0]))
            doc_url = col[2].find_all('a', href=True)[0]['href']  # retrieve the URL
            doc_url = SEC_PREFIX + doc_url  # append to the prefix
            docs.append(doc_url)
    return docs

def get_index_html_by_ticker_from_xml(ticker, form='8-K'):
    """
    Parse the filings XML. It seems XML can only retrieve 2000 records at most.
    :param ticker:
    :param form: String of form name. Default is 8-K
    :return: a dict that has filing date as key and index URL stored in a List
    """
    start = 0  # the starting record is 0 and then increased by 100
    filing_dict = dict()
    while True:
        results = get_result_from_xml(ticker, 'results', start, form)
        if not results:
            break

        for f in results[0].findall('filing'):
            date = f.find('dateFiled').text
            if date not in filing_dict:
                filing_dict[date] = []
            filing_dict[date].append(f.find('filingHREF').text)
        start += 100
    return filing_dict

def retrieve_company_doc(ticker, form='8-K', type='EX-2.1'):
    filing_dict = get_index_html_by_ticker_from_xml(ticker, form)
    doc_dict = dict()
    for date, index_pages in filing_dict.iteritems():
        docs = list(itertools.chain.from_iterable(
            [retrieve_doc_by_type(index_url, type=type) for index_url in index_pages]))
        # docs can be empty [] if no filing type (ex. EX-2.1) are found
        if docs:
            if date not in doc_dict:
                doc_dict[date] = []
                doc_dict[date] += docs
    return doc_dict

def filter_doc(doc_dict, keyword):
    """
    Loop through the dict and retrieve the docs that have the keyword.
    :param doc_dict: dict (key: date in string, value: list of doc url)
    :param keyword: string
    :return: dict (key: date in string, value: list of doc url)
    """
    new_dict = dict()
    for date, docs in doc_dict.iteritems():
        for doc in docs:
            # no explicit EX-2.1 doc is given, example: https://www.sec.gov/Archives/edgar/data/804328/000109581100000552/0001095811-00-000552-index.html
            if doc.endswith('/'):
                serial_num = doc.split('/')[-2]
                # work with the Complete submission text file
                doc = doc + serial_num[:10] + '-' + serial_num[10:12] + '-' + serial_num[12:] + '.txt'
            html = get_html(doc).read()
            if re.search(keyword, html, re.IGNORECASE):
                if date not in new_dict:
                    new_dict[date] = []
                new_dict[date].append(doc)
    return new_dict

def write_to_disk(dir, filename, content, mode='wb'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, filename), mode) as temp_file:
        writer = csv.writer(temp_file)
        if isinstance(content, dict):
            for key, value in content.items():
                writer.writerow([key, value])
        elif isinstance(content, list):
            for item in content:
                writer.writerow(item)
        elif isinstance(content, str):
            writer.writerow([content])
        else:
            writer.writerow(content)

def download_all_files(dir, doc_dict):
    for date, docs in doc_dict.iteritems():
        for doc in docs:
            content = get_html(doc).read()
            write_to_disk(dir + '/' + date, doc.split('/')[-1], content)

def read_from_csv(file, header=False):
    companies = dict()
    with open(file, 'rb') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        if not header:
            reader.next()  # skipping header
        for row in reader:
            companies[row[0]] = row[1]
    return companies

def convert_timedelta_to_hour_min_sec(t):
    """
    convert the timedelta object to a string representation in Hour, Minute, Second.
    :param t: timedelta object
    :return: String
    """
    timestr = '%s h %s min %s sec' % (str(t.seconds/3600), str((t.seconds/60)%60),
                                                  str(t.seconds % 60 + round(t.microseconds/1000000., 5)))
    return timestr

def scrape_doc_from_sec(tickers, form='8-K', type='EX-2.1', base_dir='./doc/'):
    """
    Scrape the docs with form (default=8-K) and type (default=EX-2.1) from SEC by ticker/CIK.
    :param tickers: List of string. It could be ticker or CIK
    :param form: string, default = 8-K
    :param type: string, default = EX-2.1
    :param base_dir: string, default = ./doc/
    :return: None
    """
    for ticker in tickers:
        try:
            start = datetime.datetime.now()
            ticker = str(ticker)
            logger.info('Retrieving forms for ticker = ' + ticker)
            doc_dict = retrieve_company_doc(ticker, form=form, type=type)

            # filter the documents that have the keyword (ex. Agreement and Plan of Merger)
            keyword = ''
            filtered_dict = filter_doc(doc_dict, keyword)
            # write the metadata to csv file
            write_to_disk(base_dir + ticker, ticker + '-' + form + '-' + type + '.csv', filtered_dict)
            write_all_files_to_disk(base_dir, 'CIK-' + form + '-' + type + '.csv', ticker, filtered_dict)
            # download_all_files(base_dir + ticker, filtered_dict)  # save the filtered files to disk
        except Exception as e:
            logger.error('Exception occur. Ticker = ' + ticker + ', Message: ' + str(e))
        finally:
            end = datetime.datetime.now()
            logger.info('Total time used: %s for ticker/CIK: %s'
                        % (str(convert_timedelta_to_hour_min_sec(end - start)), ticker))

def write_all_files_to_disk(dir, filename, ticker, content):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(os.path.join(dir, filename)):
        with open(os.path.join(dir, filename), 'ab') as temp_file:
            writer = csv.writer(temp_file)
            writer.writerow(['CIK/Ticker', 'Date', 'Doc URL'])
    with open(os.path.join(dir, filename), 'ab') as temp_file:
        writer = csv.writer(temp_file)
        for date, urls in content.items():
            for url in urls:
                writer.writerow([ticker, date, url])


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1





################################################################################################################
# this part for yahoo company names and SEC company names matching but it is a bit fuzzy
def cleanup_yahoo_download_file():
    yahoo_df = pd.read_csv('./Yahoo-ticker-symbol-downloader-master/stocks.csv', header=0)
    nonull_yahoo_df = yahoo_df[pd.notnull(yahoo_df['Name'])]
    nonull_yahoo_df = nonull_yahoo_df.sort_values('Name')
    yahoo_names = nonull_yahoo_df['Name']
    # Normalize the company names
    yahoo_names = cleanup_name(yahoo_names)


    # names[names.isin(['surgutneftegas open joint stock company'])]
    sec_df = pd.read_csv('./sic_cik_2017-11-09.csv', header=0)
    nonull_sec_df = sec_df[pd.notnull(sec_df['Company'])]
    nonull_sec_df = nonull_sec_df.sort_values('Company')
    sec_names = nonull_sec_df['Company']
    # Normalize the company names
    sec_names = cleanup_name(sec_names)

    file = open("./sic_cik_yahoo_search.csv", "ab")
    count = 0
    for i, name in sec_names.iteritems():
        res = find_closest_name(name, yahoo_names)[1]
        file.write(sec_df[sec_df.index == i]['Company'].values[0])
        file.write('|')
        file.write(str(res))
        file.write('\n')
        count += 1
        if count % 1000 == 0:
            print count
    file.close()


def cleanup_name(names):
    # Normalize the company names
    remove_bag_of_words = ['"', '.', '&']
    names = names.str.lower()
    for n in names:
        n = cleanco(n)
    for word in remove_bag_of_words:
        names = names.str.replace(word, '')
    names = names.str.replace(r'([^\s\w]|_)+', ' ')
    names = names.str.strip()
    names = names.drop_duplicates()
    return names

def find_closest_name(name, yahoo_names):
    distance = 999999999
    similar_names = []
    short_name = name
    for yahoo_name in yahoo_names:
        if Levenshtein.distance(short_name.split()[0], yahoo_name.split()[0]) == 0:
            new_dist = Levenshtein.distance(short_name, yahoo_name)
            if new_dist == distance:
                similar_names.append(yahoo_name)
            elif new_dist < distance:
                distance = new_dist
                similar_names = []
                similar_names.append(yahoo_name)
    return distance, similar_names
################################################################################################################

if __name__ == '__main__':
    logging.config.fileConfig('C:\Users\kcheng\PycharmProjects\\sec\logging.config')
    logger = logging.getLogger(__name__)
    start = datetime.datetime.now()

    # 1) scrape all the CIK from sec.gov by searching through all the SIC
    today = str(start.date())
    # sic_cik_file = './sic_cik_%s.csv' % today
    sic_cik_file = './sic_cik_2017-11-09.csv'
    # get_all_sic_cik_from_sec(sic_cik_file)

    # 2) scrape all the CIK for tickers that have 2 letters
    # scrape_cik_by_ticker_re(length=2)
    # scrape all the cik for tickers that have 1-5 letters
    # scrape_cik_by_ticker_xml(length=5)

    # 3) scrape yahoo finance to get ticker by CIK
    sic_cik_df = pd.read_csv(sic_cik_file, header=0)
    # company_names = sic_cik_df['Company']
    # company_names = company_names[:1]
    # scrape_ticker_by_company_name(company_names)  # TODO doesnt work

    # 4) scrape and download form and type for all tickers
    cik = sic_cik_df['CIK']
    scrape_doc_from_sec(cik, type='EX-2.[0-9]+')



    end = datetime.datetime.now()
    logger.info(convert_timedelta_to_hour_min_sec(end - start))