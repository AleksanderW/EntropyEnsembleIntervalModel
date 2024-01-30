### This project is licensed under BSD3 Licence - see the [LICENSE](LICENSE) file for details.

### Orders

**The Xu and Yager order** $[\underline{x},\overline{x}] \\leq\_{XY} [\underline{y},\overline{y}]$ if and only if:

```math
\underline{x}+\overline{x} < \underline{y}+\overline{y}\ \vee\ (\overline{x}+ \underline{x} = \overline{y} + \underline{y}, \ \overline{x} - \underline{x} \leqslant \overline{y} - \underline{y} )$
```

**The first lexicographical order** $[\underline{x},\overline{x}] \\leq\_{\\text{Lex}1} [\underline{y},\overline{y}]$ if and only if:

```math
\underline{x} < \underline{y}\ \vee\ (\underline{x} =\underline{y}, \ \overline{x} \leq \overline{y})$
```

**The second lexicographical order** $[\underline{x},\overline{x}] \\leq\_{\\text{Lex}2} [\underline{y},\overline{y}]$ if and only if:

```math
\overline{x} < \overline{y}\ \vee\ (\overline{x} =\overline{y}, \ \underline{x} \leq \underline{y})
```

### Aggregations

```math
\begin{equation}
    \mathcal{A}_{1}(\textbf{$x$}_1,\textbf{$x$}_2,...,\textbf{$x$}_n)=\left[\frac{\underline{x}_1+\underline{x}_2+...+\underline{x}_n}{n},
        \frac{\overline{x}_1+\overline{x}_2+...+\overline{x}_n}{n}\right],
\end{equation}
```

```math
\begin{equation}
    \mathcal{A}_{2}({x}_1,{x}_2,...,{x}_n) =
        \left[\frac{\underline{x}_1+\underline{x}_2+...+\underline{x}_n}{n},
        \max \left(\frac{\underline{x}_1+ \overline{x}_2+...+\overline{x}_n}{n},...,
        \frac{\overline{x}_1+\dots+\overline{x}_{n-1}+\underline{x}_n}{n}\right)\right],
\end{equation}
```

```math
\begin{equation}
    \mathcal{A}_{3}(\textbf{$x$}_1,\textbf{$x$}_2,...,\textbf{$x$}_n)=\left[\frac{\underline{x}_1+...+\underline{x}_n}{n},
        \frac{\overline{x}_1^2+...+\overline{x}_n^2}{\overline{x}_1+...+\overline{x}_n}\right],
\end{equation}
```

```math
\begin{equation}
    \mathcal{A}_{4}(\textbf{$x$}_1,...,\textbf{$x$}_n)=\left[\frac{\underline{x}_1+...+\underline{x}_n}{n},
        \frac{\overline{x}_1^3+...+\overline{x}_n^3}{\overline{x}_1^{2}+...+\overline{x}_n^{2}}\right].
\end{equation}
```

```math
\begin{equation}
    \mathcal{A}_{5}(\textbf{$x$}_1,...,\textbf{$x$}_n)=\left[\sqrt{\frac{\underline{x}_1^2+...+\underline{x}_n^2}{n}},
        \sqrt[3]{\frac{\overline{x}_1^3+...+\overline{x}_n^3}{n}}\right],
\end{equation}
```

```math
\begin{equation}
    \mathcal{A}_{6}(\textbf{$x$}_1,...,\textbf{$x$}_n)=\left[\sqrt[3]{\frac{\underline{x}_1^3+...+\underline{x}_n^3}{n}},
        \sqrt[4]{\frac{\overline{x}_1^4+...+\overline{x}_n^4}{n}}\right],
\end{equation}
```

```math
\begin{equation}
    \mathcal{A}_{7}({x}_1,{x}_2,\dots,{x}_n)=
        \left[\min \left(\frac{\overline{x}_1+ \underline{x}_2+\dots+\underline{x}_n}{n},...,
        \frac{\underline{x}_1+\dots+\underline{x}_{n-1}+\overline{x}_n}{n}\right),
        \frac{\overline{x}_1+\overline{x}_2+\dots+\overline{x}_n}{n}\right],
\end{equation}
```

```math
\begin{equation}
    \mathcal{A}_{8}(\textbf{$x$}_1,...,\textbf{$x$}_n)=\left[\sqrt[n]{\underline{x}_1 \cdot ... \cdot \underline{x}_n},
        \frac{\overline{x}_1^2+...+\overline{x}_n^2}{\overline{x}_1+...+\overline{x}_n}\right],
\end{equation}
```

```math
\begin{equation}
    \mathcal{A}_{9}(\textbf{$x$}_1,...,\textbf{$x$}_n)=\left[\sqrt{\frac{\underline{x}_1^2+...+\underline{x}_n^2}{n}},
        \frac{\overline{x}_1^3+...+\overline{x}_n^3}{\overline{x}_1^2+...+\overline{x}_n^2}\right],
\end{equation}
```

```math
\begin{equation}
    \mathcal{A}_{10}(\textbf{$x$}_1,...,\textbf{$x$}_n)=\left[\sqrt{\frac{\underline{x}_1^2+...+\underline{x}_n^2}{n}},
        \sqrt{\frac{\overline{x}_1^2+...+\overline{x}_n^2}{n}}\right].
\end{equation}
``` 
 
