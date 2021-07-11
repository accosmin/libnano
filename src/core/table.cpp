#include <nano/core/table.h>

using namespace nano;

std::ostream& nano::operator<<(std::ostream& os, const table_t& table)
{
    std::vector<size_t> colsizes(table.cols(), 0);

    // size of the value columns (in characters) - step1: single column cells
    for (const auto& row : table.content())
    {
        size_t icol = 0;
        for (const auto& cell : row.cells())
        {
            const auto span = cell.m_span;
            if (span == 1)
            {
                const auto size = cell.m_data.size() + cell.m_mark.size();
                for (size_t c = 0; c < span; ++ c, ++ icol)
                {
                    colsizes[icol] = std::max(colsizes[icol], idiv(size, span));
                }
            }
            else
            {
                icol += span;
            }
        }
    }

    // size of the value columns (in characters) - step2: make room for large multi column cells
    for (const auto& row : table.content())
    {
        size_t icol = 0;
        for (const auto& cell : row.cells())
        {
            const auto span = cell.m_span;
            if (span > 1)
            {
                const auto size = cell.m_data.size() + cell.m_mark.size();
                if (std::accumulate(
                    colsizes.begin() + static_cast<int>(icol),
                    colsizes.begin() + static_cast<int>(icol + span), size_t(0)) < size)
                {
                    for (size_t c = 0; c < span; ++ c, ++ icol)
                    {
                        colsizes[icol] = std::max(colsizes[icol], idiv(size, span));
                    }
                }
                else
                {
                    icol += span;
                }
            }
            else
            {
                icol += span;
            }
        }
    }

    //
    const auto print_row_delim = [&] ()
    {
        for (const auto colsize : colsizes)
        {
            os << "|" << string_t(colsize + 2, '-');
        }
        os << "|" << std::endl;
    };

    // display rows
    print_row_delim();
    for (const auto& row : table.content())
    {
        auto it = colsizes.begin();
        switch (row.type())
        {
        case row_t::mode::delim:
            print_row_delim();
            break;

        default:
            for (const auto& cell : row.cells())
            {
                const auto colspan = static_cast<std::ptrdiff_t>(cell.m_span);
                const auto colsize = std::accumulate(it, it + colspan, size_t(0));
                const auto extsize = (cell.m_span - 1) * 3;
                const auto coltext = cell.m_data + cell.m_mark;
                os << "| " << align(coltext, colsize + extsize, cell.m_alignment, cell.m_fill) << " ";
                std::advance(it, colspan);
            }
            os << "|" << std::endl;
            break;
        }
    }
    print_row_delim();

    return os;
}
