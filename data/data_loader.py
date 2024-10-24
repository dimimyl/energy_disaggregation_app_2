# /data/data_loader.py
import pandas as pd
import psycopg2
from datetime import datetime


class DataLoader:
    def __init__(self, db_config):
        self.db_config = db_config

    def fetch_data(self, houses_config):
        """
        Fetches data from the PostgreSQL database for multiple houses and returns a merged DataFrame.

        Parameters:
        - houses_config (list): List of dictionaries containing clientid, shellyid, and plug ids for each house.

        Returns:
        - merged_df (pd.DataFrame): The merged DataFrame containing the fetched data for all houses.
        """
        merged_dfs = []

        try:
            # Connect to PostgreSQL database
            print("Connecting to the database 'smartmeterlogs'...")
            connection = psycopg2.connect(**self.db_config)
            print("Connected to database 'smartmeterlogs'.")
            cursor = connection.cursor()

            # Define the date range
            start_date = datetime(2024, 10, 19)
            end_date = start_date + pd.DateOffset(days=4)

            for house in houses_config:
                clientid_value = house['clientid']
                deviceid_value = house['deviceid']
                plugid_value1 = house['plugid1']
                plugid_value2 = house['plugid2']

                # Define the columns to fetch
                columns_to_fetch_aggregate = ['clientid', 'shellyid', 'time', 'total_aprt_power']
                columns_to_fetch_device = ['clientid', 'shellyid', 'time', 'a_voltage', 'a_current', 'b_voltage',
                                           'b_current',
                                           'c_voltage', 'c_current']
                columns_to_fetch_plug1 = ['clientid', 'shellyid', 'time', 'apower']
                columns_to_fetch_plug2 = ['clientid', 'shellyid', 'time', 'apower']

                # Execute the queries
                aggregate_query = f"""
                    SELECT {', '.join(columns_to_fetch_aggregate)}
                    FROM public.shellypro3em
                    WHERE clientid = %s AND shellyid = 'aggregate'
                    AND time >= %s AND time < %s
                    ORDER BY time DESC
                """

                devices_query = f"""
                    SELECT {', '.join(columns_to_fetch_device)}
                    FROM public.shellypro3em
                    WHERE clientid = %s AND shellyid = %s
                    AND time >= %s AND time < %s
                    ORDER BY time DESC
                """

                plugs_query_ac = f"""
                    SELECT {', '.join(columns_to_fetch_plug1)}
                    FROM public.shellyplugs
                    WHERE clientid = %s AND shellyid = %s
                    AND time >= %s AND time < %s
                    ORDER BY time DESC
                """

                plugs_query_fridge = f"""
                    SELECT {', '.join(columns_to_fetch_plug2)}
                    FROM public.shellyplugs
                    WHERE clientid = %s AND shellyid = %s
                    AND time >= %s AND time < %s
                    ORDER BY time DESC
                """

                # Execute the queries for each house
                cursor.execute(aggregate_query, (clientid_value, start_date, end_date))
                aggregate_rows = cursor.fetchall()

                cursor.execute(devices_query, (clientid_value, deviceid_value, start_date, end_date))
                device_rows = cursor.fetchall()

                cursor.execute(plugs_query_ac, (clientid_value, plugid_value1, start_date, end_date))
                plug_rows_ac = cursor.fetchall()

                cursor.execute(plugs_query_fridge, (clientid_value, plugid_value2, start_date, end_date))
                plug_rows_fridge = cursor.fetchall()

                # Convert the results to pandas DataFrames
                df_aggregate = pd.DataFrame(aggregate_rows, columns=columns_to_fetch_aggregate)
                df_devices = pd.DataFrame(device_rows, columns=columns_to_fetch_device)
                df_plugs_ac = pd.DataFrame(plug_rows_ac, columns=columns_to_fetch_plug1)
                df_plugs_fridge = pd.DataFrame(plug_rows_fridge, columns=columns_to_fetch_plug2)

                # Merge the DataFrames
                merged_df = self._merge_dataframes(df_aggregate, df_devices, df_plugs_ac, df_plugs_fridge)

                # Append the merged DataFrame for the current house to the list
                merged_dfs.append(merged_df)

            # Concatenate all DataFrames from different houses into one
            final_merged_df = pd.concat(merged_dfs, ignore_index=True)
            final_merged_df.to_csv('combined_dataset.csv', index=False)

            return final_merged_df

        except (Exception, psycopg2.Error) as error:
            print(f"Error while connecting to PostgreSQL: {error}")
        finally:
            if connection:
                cursor.close()
                connection.close()
                print("PostgreSQL connection is closed")

    def _merge_dataframes(self, df_aggregate, df_devices, df_plugs_ac, df_plugs_fridge):
        """
        Merges the fetched DataFrames into one.

        Parameters:
        - df_aggregate (pd.DataFrame): DataFrame containing aggregate power data.
        - df_devices (pd.DataFrame): DataFrame containing device measurements.
        - df_plugs_ac (pd.DataFrame): DataFrame containing AC plug measurements.
        - df_plugs_fridge (pd.DataFrame): DataFrame containing fridge plug measurements.

        Returns:
        - merged_df (pd.DataFrame): The merged DataFrame containing all the data.
        """
        # Calculate apparent power for each phase
        df_devices['a_apparent_power'] = df_devices['a_voltage'] * df_devices['a_current']
        df_devices['b_apparent_power'] = df_devices['b_voltage'] * df_devices['b_current']
        df_devices['c_apparent_power'] = df_devices['c_voltage'] * df_devices['c_current']

        # Select only necessary columns from each DataFrame
        df_aggregate_selected = df_aggregate[['time', 'clientid', 'total_aprt_power']]
        df_devices_selected = df_devices[['time', 'a_apparent_power', 'b_apparent_power', 'c_apparent_power']]
        df_plugs_ac_selected = df_plugs_ac[['time', 'apower']]
        df_plugs_fridge_selected = df_plugs_fridge[['time', 'apower']]

        # Merging DataFrames
        merged_df = df_aggregate_selected.merge(df_devices_selected, on='time', how='outer') \
            .merge(df_plugs_ac_selected, on='time', how='outer') \
            .merge(df_plugs_fridge_selected, on='time', how='outer')

        # Renaming columns according to specified rules
        merged_df.rename(columns={
            'total_aprt_power': 'agg',
            'a_apparent_power': 'wm',
            'b_apparent_power': 'st',
            'c_apparent_power': 'wh',
            'apower_x': 'ac_power',
            'apower_y': 'fridge_power'
        }, inplace=True)

        return merged_df



