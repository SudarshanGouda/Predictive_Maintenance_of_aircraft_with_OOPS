from ProcessongandCleaning import *

if __name__ == '__main__':

    ### Loading New Data and Predicting

    model = airclaft_predictive_model('E:\DATA Science\pythonProject\Predictive Maintenance of aircraft\model')

    model.load_clean_data('E:\DATA Science\pythonProject\Predictive Maintenance of aircraft\PM_train.txt')

    presicted_df = model.predicted_outputs()

    ### Checking the result

    df_act = pd.read_csv('./PM_truth.txt', sep=' ', header=None)

    df_act.drop(1, axis=1, inplace=True)
    df_act.columns = ['Actal']

    Table = pd.concat([presicted_df, df_act], axis=1)

    print(Table.head())
