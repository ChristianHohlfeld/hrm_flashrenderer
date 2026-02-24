#include "hrm/signature.h"
#include "hrm/common.h"
#include <stdexcept>

static void assert_true(bool v, const char* msg) {
    if (!v) throw std::runtime_error(msg);
}

void test_signature() {
    auto q1 = hrm::qbins_from_text("Romeo Romeo wherefore art thou Romeo");
    auto q2 = hrm::qbins_from_text("romoe romeo wherefore art thou romeo"); // typo
    assert_true(q1.size() == hrm::BINS, "qbins size");
    assert_true(q2.size() == hrm::BINS, "qbins size 2");

    int self = 0, typo = 0, other = 0;
    for (int i = 0; i < hrm::BINS; i++) {
        self += hrm::imin(q1[i], q1[i]);
        typo += hrm::imin(q1[i], q2[i]);
    }
    auto q3 = hrm::qbins_from_text("completely unrelated query about physics");
    for (int i = 0; i < hrm::BINS; i++) other += hrm::imin(q1[i], q3[i]);

    assert_true(self > 0, "nonzero signature");
    assert_true(self >= typo, "self >= typo");
    assert_true(typo >= other, "typo >= other (locality)");
}
