# JuliaでLogisticRegression
参考サイト  
- [Flux-LogisticRegression](https://fluxml.ai/Flux.jl/stable/tutorials/logistic_regression/)

## 環境
- PC: MacBook Air M2, 2022
- OS: macOS Sonoma 14.1.1
- Julia: 1.9.3

## ディレクトリ構造
Juliaのproject機能で管理  
```
LogisticRegression_julia
├── Manifest.toml
├── Project.toml
├── doc
│   └── Readme.md
└── src
    ├── LogisticRegression.jl
    └── main.jl
```

## 実行方法
※ Juliaはインストール済かつVS Codeで環境構築してある前提  
### VS Codeで実行
1. VS Codeで`LogisticRegression_julia`ディレクトリを開く
2. `LogisticRegression.jl`ファイルを開き，右上の▶️でコンパイル
3. `main.jl`ファイルを開き，同様にコンパイル
4. コンパイルした際に`julia REPL`が起動するので，そこで`main()`を実行
5. パラメータの情報を確認したい場合は，`params = main()`などで実行すると，そのままREPLにて`params.W`や`params.b`で参照できる