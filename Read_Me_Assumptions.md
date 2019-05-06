
<h2><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a> Assumptions
</h2>

In this serie of computation, two datasets will be used to perform two regressions analysis. 
We consider the following assumptions to be validated :
- Linear relationship : the relationship between the independent and dependent variables needs to be linear.
- Multivariate normality : observations are normaly distributed 
- Indepent error : implies no correlations in the errors (residuals)
- Homoscedasticity: the residuals accros the regression line are aquls


1. Given any two crude oils with their given distillation profiles, create a model which will give an approximate distillation
profile of the mixture of the two oils with specified volumes. [...]

The selected model for this case is a simple linear model. Respectively follow the proprieties of:

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/441d157d1b9e322b3cf27b721a370be6844d30c8" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.671ex; width:18.122ex; height:2.509ex;" alt="{\displaystyle y=\beta _{0}+\beta _{1}x+\varepsilon ,\,}">


2.Collect data from Crude Monitor for a couple of real crudes, and cleaning recent data, run the distillation
profiles through your program with volumes of your choosing.

we choosed to use a more complexe model to illustrate the non peferctly linear relationship between the two variables.
Respectively the general polynomial model follows :

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/ba2b6f48bb60ea6fe6865a81956146142fdac62a" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.671ex; width:25.716ex; height:3.009ex;" alt="{\displaystyle y=\beta _{0}+\beta _{1}x+\beta _{2}x^{2}+\varepsilon .\,}">

The dataset used for this computation is Crude-file.csv.
All computing and results are available in pdf.
