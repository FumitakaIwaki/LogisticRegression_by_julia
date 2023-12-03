module LogReg
    using Flux, Statistics, MLDatasets, DataFrames, OneHotArrays

    # パラメータの構造体
    mutable struct Params
        n_targets::Int64 # クラス数
        n_features::Int64 # 特徴量数
        classes::Vector{Any} # クラス名
        W::Array{Float32} # 重み
        b::Vector{Float32} # バイアス
        custom_y_onehot::BitMatrix # クラスのonehot

        Params() = new()
    end

    params = Params(); # パラメータコンストラクタの作成


    # モデルのセットアップ
    function init_model(x, y)
        params.n_targets = length(unique(y));
        params.n_features = size(x)[1];
        params.classes = Vector();
        for i = 1:params.n_targets
            append!(params.classes, [unique(y)[i]]);
        end
        params.custom_y_onehot = unique(y) .== permutedims(y);
        params.W = rand(Float32, params.n_targets, params.n_features);
        params.b = zeros(Float32, params.n_targets);
    end


    # 演算
    function m(W, b, x)
        return W*x .+ b;
    end


    # ソフトマックス関数
    function custom_softmax(x)
        return exp.(x) ./ sum(exp.(x), dims=1);
    end


    # モデルの予測値を算出する関数
    function custom_model(W, b, x)
        return custom_softmax(m(W, b, x));
    end


    # 交差エントロピー誤差関数
    function  custom_logitcrossentropy(y_hat, y)
        return mean(.-sum(y .* logsoftmax(y_hat; dims = 1); dims = 1));
    end


    # モデルで予測値を出力し，その誤差を算出する関数
    function custom_loss(W, b, x, y)
        y_hat = custom_model(W, b, x);
        return custom_logitcrossentropy(y_hat, y);
    end


    # 最も予測値の高いラベルに分類する関数
    function custom_onecold(custom_y_onehot)
        max_idx = [x[1] for x in argmax(custom_y_onehot; dims = 1)];
        return vec(params.classes[max_idx]);
    end


    # 精度を算出する関数
    function custom_accuracy(W, b, x, y)
        return mean(custom_onecold(custom_model(W, b, x)) .== y);
    end


    # パラメータの更新
    function train_custom_model(W, b, x, custom_y_onehot)
        dLdW, dLdb, _, _ = gradient(custom_loss, W, b, x, custom_y_onehot);
        W .= W .- 0.1 .* dLdW;
        b .= b .- 0.1 .* dLdb;

        return W, b
    end


    # 訓練の実行
    function train(x, y)
        for i = 1:1000
            # 学習により重みとバイアスを更新
            params.W, params.b = train_custom_model(params.W, params.b, x, params.custom_y_onehot);
            # 精度が0.98を超えたら強制終了
            custom_accuracy(params.W, params.b, x, y) >= 0.98 && break;
        end
        
        return params
    end

end 
