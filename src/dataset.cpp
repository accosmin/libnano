#include <mutex>
#include <nano/logger.h>
#include <nano/mlearn/class.h>
#include <nano/dataset/tabular.h>
#include <nano/dataset/imclass_cifar.h>
#include <nano/dataset/imclass_mnist.h>

using namespace nano;

static bool is_multi_class(const tensor3d_cmap_t& targets)
{
    return std::count_if(begin(targets), end(targets), [] (scalar_t value) { return is_pos_target(value); }) != 1;
}

task_type dataset_t::type() const
{
    const auto target = this->target();

    if (!target)
    {
        return task_type::unsupervised;
    }
    else
    {
        critical(
            target.optional(),
            scat("dataset: the target feature (", target.name(), ") cannot be optional!"));

        if (!target.discrete())
        {
            return task_type::regression;
        }
        else
        {
            // decide if single-label or multi-label
            bool multi_class = false;
            const auto batch = tensor_size_t{1024};
            for (tensor_size_t begin = 0, samples = this->samples(); begin < samples && !multi_class; begin += batch)
            {
                const auto end = std::min(begin + batch, samples);
                const auto targets = this->targets(arange(begin, end));
                for (tensor_size_t i = 0; i < targets.size<0>() && !multi_class; ++ i)
                {
                    multi_class = is_multi_class(targets.tensor(i));
                }
            }

            return multi_class ? task_type::mclassification : task_type::sclassification;
        }
    }
}

dataset_factory_t& dataset_t::all()
{
    static dataset_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/");

        manager.add<tabular_dataset_t>(
            "iris",
            "classify flowers from physical measurements of the sepal and petal (Fisher, 1936)",
            csvs_t
            {
                csv_t{dir + "/iris/iris.data"}.delim(",").header(false).expected(150)
            },
            features_t
            {
                feature_t{"sepal_length_cm"},
                feature_t{"sepal_width_cm"},
                feature_t{"petal_length_cm"},
                feature_t{"petal_width_cm"},
                feature_t{"class"}.discrete(3),
            }, 4);

        manager.add<tabular_dataset_t>(
            "wine",
            "predict the wine type from its constituents (Aeberhard, Coomans & de Vel, 1992)",
            csvs_t
            {
                csv_t{dir + "/wine/wine.data"}.delim(",").header(false).expected(178)
            },
            features_t
            {
                feature_t{"class"}.discrete(3),
                feature_t{"Alcohol"},
                feature_t{"Malic acid"},
                feature_t{"Ash"},
                feature_t{"Alcalinity of ash"},
                feature_t{"Magnesium"},
                feature_t{"Total phenols"},
                feature_t{"Flavanoids"},
                feature_t{"Nonflavanoid phenols"},
                feature_t{"Proanthocyanins"},
                feature_t{"Color intensity"},
                feature_t{"Hue"},
                feature_t{"OD280/OD315 of diluted wines"},
                feature_t{"Proline"},
            }, 0);

        manager.add<tabular_dataset_t>(
            "adult",
            "predict if a person makes more than 50K per year (Kohavi & Becker, 1994)",
            csvs_t
            {
                csv_t{dir + "/adult/adult.data"}.skip('|').delim(", .").header(false).expected(32561),
                csv_t{dir + "/adult/adult.test"}.skip('|').delim(", .").header(false).expected(16281).testing(make_range(0, 16281))
            },
            features_t
            {
                feature_t{"age"},
                feature_t{"workclass"}.placeholder("?").discrete(8),
                feature_t{"fnlwgt"},
                feature_t{"education"}.discrete(16),
                feature_t{"education-num"},
                feature_t{"marital-status"}.discrete(7),
                feature_t{"occupation"}.placeholder("?").discrete(14),
                feature_t{"relationship"}.discrete(6),
                feature_t{"race"}.discrete(5),
                feature_t{"sex"}.discrete({"Female", "Male"}),
                feature_t{"capital-gain"},
                feature_t{"capital-loss"},
                feature_t{"hours-per-week"},
                feature_t{"native-country"}.placeholder("?").discrete(41),
                feature_t{"income"}.discrete(2),
            }, 14);

        manager.add<tabular_dataset_t>(
            "abalone",
            "predict the age of abalone from physical measurements (Waugh, 1995)",
            csvs_t
            {
                csv_t{dir + "/abalone/abalone.data"}.delim(",").header(false).expected(4177).testing(make_range(3133, 4177))
            },
            features_t
            {
                feature_t{"sex"}.discrete(3),
                feature_t{"length"},
                feature_t{"diameter"},
                feature_t{"height"},
                feature_t{"whole_weight"},
                feature_t{"shucked_weight"},
                feature_t{"viscera_weight"},
                feature_t{"shell_weight"},
                feature_t{"rings"}.discrete(29),
            }, 8);

        manager.add<tabular_dataset_t>(
            "forest-fires",
            "predict the burned area of the forest (Cortez & Morais, 2007)",
            csvs_t
            {
                csv_t{dir + "/forest-fires/forestfires.csv"}.delim(",").header(true).expected(517)
            },
            features_t
            {
                feature_t{"X"}.discrete(9),
                feature_t{"Y"}.discrete(8),
                feature_t{"month"}.discrete(12),
                feature_t{"day"}.discrete(7),
                feature_t{"FFMC"},
                feature_t{"DMC"},
                feature_t{"DC"},
                feature_t{"ISI"},
                feature_t{"temp"},
                feature_t{"RH"},
                feature_t{"wind"},
                feature_t{"rain"},
                feature_t{"area"}
            }, 12);

        manager.add<tabular_dataset_t>(
            "breast-cancer",
            "diagnostic breast cancer using measurements of cell nucleai (Street, Wolberg & Mangasarian, 1992)",
            csvs_t
            {
                csv_t{dir + "/breast-cancer/wdbc.data"}.delim(",").header(false).expected(569)
            },
            features_t
            {
                feature_t{"ID"},
                feature_t{"Diagnosis"}.discrete(2),

                feature_t{"radius1"},
                feature_t{"texture1"},
                feature_t{"perimeter1"},
                feature_t{"area1"},
                feature_t{"smoothness1"},
                feature_t{"compactness1"},
                feature_t{"concavity1"},
                feature_t{"concave_points1"},
                feature_t{"symmetry1"},
                feature_t{"fractal_dimension1"},

                feature_t{"radius2"},
                feature_t{"texture2"},
                feature_t{"perimeter2"},
                feature_t{"area2"},
                feature_t{"smoothness2"},
                feature_t{"compactness2"},
                feature_t{"concavity2"},
                feature_t{"concave_points2"},
                feature_t{"symmetry2"},
                feature_t{"fractal_dimension2"},

                feature_t{"radius3"},
                feature_t{"texture3"},
                feature_t{"perimeter3"},
                feature_t{"area3"},
                feature_t{"smoothness3"},
                feature_t{"compactness3"},
                feature_t{"concavity3"},
                feature_t{"concave_points3"},
                feature_t{"symmetry3"},
                feature_t{"fractal_dimension3"}
            }, 1);

        manager.add<tabular_dataset_t>(
            "bank-marketing",
            "predict if a client has subscribed a term deposit (Moro, Laureano & Cortez, 2011)",
            csvs_t
            {
                csv_t{dir + "/bank-marketing/bank-additional-full.csv"}.delim(";\"\r").header(true).expected(41188)
            },
            features_t
            {
                feature_t{"age"},
                feature_t{"job"}.discrete(12),
                feature_t{"marital"}.discrete(4),
                feature_t{"education"}.discrete(8),
                feature_t{"default"}.discrete(3),
                feature_t{"housing"}.discrete(3),
                feature_t{"loan"}.discrete(3),
                feature_t{"contact"}.discrete(2),
                feature_t{"month"}.discrete(12),
                feature_t{"day_of_week"}.discrete(5),
                feature_t{"duration"},
                feature_t{"campaign"},
                feature_t{"pdays"},
                feature_t{"previous"},
                feature_t{"poutcome"}.discrete(3),
                feature_t{"emp.var.rate"},
                feature_t{"cons.price.idx"},
                feature_t{"cons.conf.idx"},
                feature_t{"euribor3m"},
                feature_t{"nr.employed"},
                feature_t{"y"}.discrete(2),
            }, 20);

        manager.add<mnist_dataset_t>("mnist",
            "classify 28x28 grayscale images of hand-written digits (MNIST)");
        manager.add<cifar10_dataset_t>("cifar10",
            "classify 3x32x32 color images (CIFAR-10)");
        manager.add<cifar100c_dataset_t>("cifar100c",
            "classify 3x32x32 color images (CIFAR-100 with 20 coarse labels)");
        manager.add<cifar100f_dataset_t>("cifar100f",
            "classify 3x32x32 color images (CIFAR-100 with 100 fine labels)");
        manager.add<fashion_mnist_dataset_t>("fashion-mnist",
            "classify 28x28 grayscale images of fashion articles (Fashion-MNIST)");
    });

    return manager;
}
