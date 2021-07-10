#include <utest/utest.h>
#include <nano/core/tokenizer.h>

UTEST_BEGIN_MODULE(test_core_tokenizer)

UTEST_CASE(split_str)
{
    const auto str = std::string{"= -token1 token2 something "};
    for (auto tokenizer = nano::tokenizer_t{str, " =-"}; tokenizer; tokenizer ++)
    {
        switch (tokenizer.count())
        {
        case 1:     UTEST_CHECK_EQUAL(tokenizer.get(), "token1");
                    UTEST_CHECK_EQUAL(tokenizer.pos(), 3);
                    break;

        case 2:     UTEST_CHECK_EQUAL(tokenizer.get(), "token2");
                    UTEST_CHECK_EQUAL(tokenizer.pos(), 10);
                    break;

        case 3:     UTEST_CHECK_EQUAL(tokenizer.get(), "something");
                    UTEST_CHECK_EQUAL(tokenizer.pos(), 17);
                    break;

        default:    UTEST_CHECK(false);
        }
    }
}

UTEST_CASE(split_char)
{
    const auto str = std::string{"= -token1 token2 something"};
    for (auto tokenizer = nano::tokenizer_t{str, "-"}; tokenizer; ++ tokenizer)
    {
        switch (tokenizer.count())
        {
        case 1:     UTEST_CHECK_EQUAL(tokenizer.get(), "= ");
                    UTEST_CHECK_EQUAL(tokenizer.pos(), 0);
                    break;

        case 2:     UTEST_CHECK_EQUAL(tokenizer.get(), "token1 token2 something");
                    UTEST_CHECK_EQUAL(tokenizer.pos(), 3);
                    break;

        default:    UTEST_CHECK(false);
        }
    }
}

UTEST_CASE(split_none)
{
    const auto str = std::string{"= -token1 token2 something "};
    for (auto tokenizer = nano::tokenizer_t{str, "@"}; tokenizer; ++ tokenizer)
    {
        switch (tokenizer.count())
        {
        case 1:     UTEST_CHECK_EQUAL(tokenizer.get(), "= -token1 token2 something ");
                    break;

        default:    UTEST_CHECK(false);
        }
    }
}

UTEST_END_MODULE()
