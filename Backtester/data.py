#!/usr/bin/python
# -*- coding: utf-8 -*-
# data.py

from abc import ABCMeta, abstractmethod
import datetime
import os, os.path
import numpy as np
import pandas as pd
from event import MarketEvent


class DataHandler(object):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).
    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OHLCVI) for each symbol requested.
    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_symbol_list(self):
        """
        Returns the last bar updated.
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bar(self, symbol):
        """
        Returns the last bar updated.
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        from the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        raise NotImplementedError("Should implement get_latest_bars_values()")

    @abstractmethod
    def update_bars(self, events):
        """
        Pushes the latest bars to the bars_queue for each symbol
        in a tuple OHLCVI format: (datetime, open, high, low,
        close, volume, open interest).
        """
        raise NotImplementedError("Should implement update_bars()")


class dfDataHandler(DataHandler):

    def __init__(self):
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.comb_index = None
        self.continue_backtest = True

    def get_symbol_list(self):
        return self.symbol_data.keys()

    def AddSymbolData(self, symbol, df):
        self.symbol_data[symbol] = df

        # Combine the index to pad forward values
        if self.comb_index is None:
            self.comb_index = self.symbol_data[symbol].index
        else:
            self.comb_index.union(self.symbol_data[symbol].index)

        # Set the latest symbol_data to None
        self.latest_symbol_data[symbol] = []

    def Finish(self):
        # Reindex the dataframes
        for s in self.symbol_data.keys():
            self.symbol_data[s] = self.symbol_data[s]. \
                reindex(index=self.comb_index, method='pad').iterrows()

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """

        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """

        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """

        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """

        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def appendToLatest(self,  symbol, bar):
        self.latest_symbol_data[symbol].append(bar)

    def update_bars(self, events):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """

        for s in self.symbol_data.keys():
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.appendToLatest(s, bar)
        events.put(MarketEvent())


class HistoricDatabaseDataHandler(dfDataHandler):
    """
    HistoricCSVDataHandler is designed to read securities master database for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """

    def __init__(self):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        ’symbol.csv’, where symbol is a string in the list.
        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """

        super(HistoricDatabaseDataHandler, self).__init__()
        self.symbol_list = []
        self._readFromDB()

    def _readFromDB(self):
        """
        get data from the securities master database
        """
        import mysql.connector
        from sqlalchemy import create_engine

        def getDataFor(ticker):
            sql = """SELECT dp.price_date, dp.adj_close_price
            FROM symbol AS sym
            INNER JOIN daily_price AS dp
            ON dp.symbol_id = sym.id
            WHERE sym.ticker = '{}'
            ORDER BY dp.price_date ASC;""".format(ticker)
            # Create a pandas dataframe from the SQL query
            engine = create_engine('mysql+mysqlconnector://sec_user:password@localhost/securities_master', echo=False)
            with engine.connect() as conn, conn.begin():
                return pd.read_sql_query(sql, con=conn, index_col='price_date')

        engine = create_engine('mysql+mysqlconnector://sec_user:password@localhost/securities_master', echo=False)

        with engine.connect() as con:
            data = con.execute("SELECT ticker FROM symbol")

        self.symbol_list = [d['ticker'] for d in data]

        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.AddSymbolData(s, getDataFor(s))

        self.Finish()


class HistoricCSVDataHandler(dfDataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """

    def __init__(self, events, csv_dir, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        ’symbol.csv’, where symbol is a string in the list.
        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """

        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.continue_backtest = True
        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        For this handler it will be assumed that the data is
        taken from Yahoo. Thus its format will be respected.
        """

        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.AddSymbolData(s, pd.io.parsers.read_csv(
                os.path.join(self.csv_dir, ' % s.csv' % s),
                header=0, index_col=0, parse_dates=True,
                names = [ 'datetime', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
            ).sort())

        self.Finish()
