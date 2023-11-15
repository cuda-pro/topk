#include <raft/sparse/detail/utils.h>
#include <stdint.h>
#include <stdlib.h>

#include <numeric>
#include <optional>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/select_k.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>

void raft_matrix_selectk(int topk, int batch_size, int n_docs,
                         const std::vector<float>& scores, const std::vector<int>& indeces,
                         std::vector<float>& s_scores, std::vector<int>& s_doc_ids) {
    raft::resources handle;
    auto stream = raft::resource::get_cuda_stream(handle);
    rmm::device_uvector<float> d_scores(scores.size(), stream);
    raft::update_device(d_scores.data(), scores.data(), scores.size(), stream);
    rmm::device_uvector<int> d_doc_ids(n_docs, stream);
    if (indeces.empty()) {
        // like std:itoa on gpu device global memory
        raft::sparse::iota_fill(d_doc_ids.data(), batch_size, int(n_docs), stream);
        // stream.synchronize();
    } else {
        raft::update_device(d_doc_ids.data(), indeces.data(), indeces.size(), stream);
    }
    rmm::device_uvector<float> d_out_scores(batch_size * topk, stream);
    rmm::device_uvector<int> d_out_ids(batch_size * topk, stream);
    auto in_extent = raft::make_extents<int64_t>(batch_size, n_docs);
    auto out_extent = raft::make_extents<int64_t>(batch_size, topk);
    auto in_span = raft::make_mdspan<const float, int64_t, raft::row_major, false, true>(d_scores.data(), in_extent);
    auto in_idx_span = raft::make_mdspan<const int, int64_t, raft::row_major, false, true>(d_doc_ids.data(), in_extent);
    auto out_span = raft::make_mdspan<float, int64_t, raft::row_major, false, true>(d_out_scores.data(), out_extent);
    auto out_idx_span = raft::make_mdspan<int, int64_t, raft::row_major, false, true>(d_out_ids.data(), out_extent);

    // note: if in_idx_span is null use std::nullopt prevents automatic inference of the template parameters.
    // raft::matrix::select_k<float, int>(handle, in_span, std::nullopt, out_span, out_idx_span, false, true);
    raft::matrix::select_k<float, int>(handle, in_span, std::optional(in_idx_span), out_span, out_idx_span, false, true);

    raft::update_host(s_scores.data(), d_out_scores.data(), d_out_scores.size(), stream);
    raft::update_host(s_doc_ids.data(), d_out_ids.data(), d_out_ids.size(), stream);
    raft::interruptible::synchronize(stream);
}

int main(int argc, char* argv[]) {
    int topk = argc > 1 ? atoi(argv[1]) : 100;
    int batch_size = 1;
    std::vector<float> scores = {0.928571, 0.9, 0.896552, 0.875, 0.875, 0.870968, 0.866667, 0.866667, 0.866667, 0.857143, 0.857143, 0.857143, 0.83871, 0.83871, 0.83871, 0.818182, 0.818182, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125, 0.806452, 0.8, 0.794118, 0.794118, 0.787879, 0.787879, 0.787879, 0.78125, 0.771429, 0.771429, 0.771429, 0.764706, 0.764706, 0.764706, 0.764706, 0.757576, 0.742857, 0.742857, 0.72973, 0.72973, 0.722222, 0.722222, 0.722222, 0.722222, 0.722222, 0.71875, 0.702703, 0.702703, 0.702703, 0.702703, 0.702703, 0.702703, 0.692308, 0.685714, 0.684211, 0.684211, 0.676471, 0.666667, 0.658537, 0.634146, 0.628571, 0.611111, 0.606061, 0.604651, 0.571429, 0.568182, 0.565217, 0.522727, 0.490566, 0.485714, 0.482759, 0.466667, 0.466667, 0.464286, 0.464286, 0.464286, 0.464286, 0.464286, 0.464286, 0.464286, 0.454545, 0.451613, 0.451613, 0.451613, 0.451613, 0.451613, 0.451613, 0.448276, 0.448276, 0.448276, 0.448276, 0.448276, 0.444444, 0.441176, 0.441176, 0.441176, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.433333, 0.433333, 0.433333, 0.433333, 0.433333, 0.433333, 0.433333, 0.433333, 0.433333, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.428571, 0.424242, 0.424242, 0.424242, 0.424242, 0.424242, 0.424242, 0.424242, 0.424242, 0.424242, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.419355, 0.416667, 0.416667, 0.416667, 0.416667, 0.416667, 0.416667, 0.416667, 0.416667, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793, 0.413793};
    size_t n_docs = scores.size();
    std::cout << "size:" << n_docs << " scores:" << std::endl;
    for (auto& s : scores) {
        // s *= -1;
        std::cout << s << ",";
    }
    std::cout << std::endl;

    // std::vector<int> indeces = {1064412, 1212301, 1165199, 1324553, 1337544, 1279350, 1221518, 1244284, 1244680, 972283, 1009627, 1094325, 1252175, 1260777, 1268224, 1384210, 1393825, 1310814, 1315547, 1317017, 1329501, 1341171, 1292801, 1477389, 1427999, 1443623, 1351931, 1352887, 1363146, 1307186, 1480921, 1490375, 1499799, 1418128, 1435011, 1439307, 1440123, 1366052, 1485951, 1486511, 1565048, 1566093, 1500616, 1511639, 1512826, 1538358, 1547424, 1318251, 1554073, 1571537, 1576178, 1588176, 1596407, 1597752, 1690975, 1484789, 1613338, 1622281, 1435168, 1661370, 1772693, 1774156, 1467924, 1527674, 1380085, 1876256, 574048, 1951930, 2078346, 1932445, 2416096, 1450838, 1182901, 1216202, 1237692, 1060174, 1086354, 1091797, 1100362, 1123673, 1133630, 1149061, 1348497, 1255132, 1255449, 1260932, 1264037, 1275180, 1288477, 1150172, 1153735, 1184612, 1184758, 1189700, 1535638, 1403109, 1403478, 1408418, 1299546, 1303931, 1305020, 1307673, 1308657, 1310970, 1313908, 1328748, 1331627, 1335584, 1337183, 1337396, 1205270, 1207494, 1210580, 1215826, 1224063, 1229376, 1233908, 1239968, 1244790, 1052359, 1056242, 1058846, 1082727, 1084014, 1086005, 1088113, 1090403, 1094766, 1095157, 1098492, 1120177, 1123313, 1125262, 1132786, 1137509, 1139087, 1139990, 1143328, 1145536, 1449801, 1450239, 1462119, 1463176, 1467099, 1473971, 1354489, 1356513, 1357420, 1372172, 1373926, 1389397, 1392205, 1395926, 1397566, 1248252, 1249709, 1253140, 1263190, 1266363, 1269704, 1283900, 1284511, 1286084, 1286746, 1287506, 1290151, 1290708, 1296362, 1510143, 1520730, 1528193, 1538214, 1541955, 1546143, 1546969, 1547567, 1149781, 1150162, 1157819, 1158335, 1160654, 1161891, 1163208, 1163223, 1164708, 1167858, 1173572, 1175021, 1178008, 1178115, 1181306, 1185345, 1186252, 1187251, 1188466, 1188505, 1189249, 1190057, 1190158, 1190416};
    std::vector<int> indeces;
    std::cout << " indeces:" << std::endl;
    for (auto i : indeces) {
        std::cout << i << ",";
    }
    std::cout << std::endl;

    std::vector<float> s_scores(batch_size * topk);
    std::vector<int> s_doc_ids(batch_size * topk);

    raft_matrix_selectk(topk, batch_size, n_docs, scores, indeces, s_scores, s_doc_ids);

    std::cout << "s_scores:" << std::endl;
    for (auto s : s_scores) {
        std::cout << s << ",";
    }
    std::cout << std::endl;
    std::cout << "s_doc_ids:" << std::endl;
    for (auto id : s_doc_ids) {
        std::cout << id << ",";
    }
    std::cout << std::endl;
}