using Flux, Statistics, MLDatasets, DataFrames, OneHotArrays
using .LogReg


# アヤメデータを呼び出す関数
function load_iris()
    x, y = Iris(as_df=false)[:];
    x = Float32.(x);
    y = vec(y);

    return x, y;
end

# main関数
function main()
    # アヤメデータ
    x, y = load_iris();
    # モデルの初期化
    LogReg.init_model(x, y)

    # train
    LogReg.train(x, y);

    # loss計算
    loss = LogReg.custom_loss(LogReg.params.W, LogReg.params.b, x, LogReg.params.custom_y_onehot);
    println("Loss: ", loss)
    # Accuracy計算
    accuracy = LogReg.custom_accuracy(LogReg.params.W, LogReg.params.b, x, y);
    println("Accuracy: ", accuracy)


    return LogReg.params
end