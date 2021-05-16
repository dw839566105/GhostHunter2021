import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
h2o.init(port = 44944, max_mem_size = "900g", nthreads = 100)
hf = h2o.import_file("Juno2_20_20.parquet")
pairs = []
for index, i in enumerate(hf.columns):
    for j in hf.columns:
        if (i.startswith('L') and j.startswith('T')):
            if ((i!='L0') and (j != 'T0')):
                pairs.append((i, j))

predictors = hf.columns[:-1]
response_col = hf.columns[-1]

print('begin...')
glm_model = H2OGeneralizedLinearEstimator(family= "poisson",
    interaction_pairs=pairs,
    lambda_ = 0,
    remove_collinear_columns = True)
    #compute_p_values = True)

glm_model.train(predictors, response_col, training_frame=hf)

coef_table = glm_model._model_json['output']['coefficients_table']

h2o.save_model(model=glm_model, path="./model3", force=True)
coef_ = coef_table['coefficients']
AIC = glm_model.aic()

print(glm_model)