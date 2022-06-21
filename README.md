# VM

**Grid Method**

Решается уравнение параболического типа методом сеток через явную и неявную схему.

**Vandermonde**

Составляется система линейных уравнений с обобщенной матрицей Вандермонда (у нее очень большое число обусловленности, а значит непозволительно ее решать напрямую), затем регуляризуется (прямым методом через сведение к самосопряженной матрице и другим методом через корень из матрицы). Как известно, решение регуляризованной системы (в ней мы можем управлять параметром альфа) сходится к решению исходной при уменьшении параметра альфа. К тому же регуляризация позволяет контролировать точность вычислений, тем самым мы обходим вопрос с большими погрешностями.  

