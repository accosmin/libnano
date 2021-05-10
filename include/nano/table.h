#pragma once

#include <limits>
#include <cassert>
#include <numeric>
#include <utility>
#include <algorithm>
#include <nano/arch.h>
#include <nano/scalar.h>
#include <nano/string.h>
#include <nano/core/numeric.h>

namespace nano
{
    ///
    /// \brief cell in a table, potentially spanning multiple columns.
    ///
    struct cell_t
    {
        ///
        /// \brief default constructor
        ///
        cell_t() = default;

        ///
        /// \brief constructor
        ///
        cell_t(string_t data, const size_t span, const alignment align, const char fill) :
            m_data(std::move(data)),
            m_span(span),
            m_fill(fill),
            m_alignment(align)
        {
        }

        // attributes
        string_t    m_data;                     ///<
        string_t    m_mark;                     ///<
        size_t      m_span{1};                  ///< column spanning
        char        m_fill{' '};                ///< filling character for aligning cells
        alignment   m_alignment{alignment::left};///< text alignment within the cell
    };

    inline bool operator==(const cell_t& c1, const cell_t& c2)
    {
        return c1.m_data == c2.m_data && c1.m_span == c2.m_span && c1.m_alignment == c2.m_alignment;
    }

    ///
    /// \brief control column spanning.
    ///
    struct colspan_t
    {
        size_t      m_span{1};
    };

    ///
    /// \brief control filling for aligning text in a cell.
    ///
    struct colfill_t
    {
        char        m_fill{' '};
    };

    inline colspan_t colspan(const size_t span) { return {span}; }
    inline colfill_t colfill(const char fill) { return {fill}; }

    ///
    /// \brief row in a table, consisting of a list of cells.
    ///
    class row_t
    {
    public:

        enum class mode
        {
            data,       ///<
            delim,      ///< delimiting row
            header,     ///< header (not considered for operations like sorting or marking)
        };

        ///
        /// \brief default constructor
        ///
        row_t() = default;

        ///
        /// \brief constructor
        ///
        explicit row_t(const mode t) :
            m_type(t)
        {
        }

        ///
        /// \brief change the current formatting to be used by the next cells
        ///
        row_t& operator<<(const alignment align) { m_alignment = align; return *this; }
        row_t& operator<<(const colfill_t colfill) { m_colfill = colfill.m_fill; return *this; }
        row_t& operator<<(const colspan_t colspan) { m_colspan = colspan.m_span; return *this; }

        ///
        /// \brief insert new cells using the current formatting settings
        ///
        row_t& operator<<(const char* string)
        {
            m_cells.emplace_back(string, m_colspan, m_alignment, m_colfill);
            m_cols += m_colspan;
            return (*this) << colspan(1) << alignment::left << colfill(' ');
        }
        row_t& operator<<(const string_t& string)
        {
            m_cells.emplace_back(string, m_colspan, m_alignment, m_colfill);
            m_cols += m_colspan;
            return (*this) << colspan(1) << alignment::left << colfill(' ');
        }
        template <typename tscalar>
        row_t& operator<<(const tscalar& value)
        {
            m_cells.emplace_back(scat(value), m_colspan, m_alignment, m_colfill);
            m_cols += m_colspan;
            return (*this) << colspan(1) << alignment::left << colfill(' ');
        }
        template <typename tscalar>
        row_t& operator<<(const std::vector<tscalar>& values)
        {
            for (const auto& value : values)
            {
                operator<<(value);
            }
            return *this;
        }

        ///
        /// \brief find the a cell taking into account column spanning
        ///
        cell_t* find(const size_t col)
        {
            size_t icol = 0;
            const auto it = std::find_if(m_cells.begin(), m_cells.end(), [&] (const auto& cell)
            {
                icol += cell.m_span; return icol > col;
            });
            return (it == m_cells.end()) ? nullptr : &*it;
        }

        const cell_t* find(const size_t col) const
        {
            size_t icol = 0;
            const auto it = std::find_if(m_cells.begin(), m_cells.end(), [&] (const auto& cell)
            {
                icol += cell.m_span; return icol > col;
            });
            return (it == m_cells.end()) ? nullptr : &*it;
        }

        ///
        /// \brief change a column's mark or data (finds the right cell taking into account column spanning)
        ///
        void data(const size_t col, const string_t& str) { auto* cell = find(col); assert(cell); cell->m_data = str; }
        void mark(const size_t col, const string_t& str) { auto* cell = find(col); assert(cell); cell->m_mark = str; }

        ///
        /// \brief collect the columns as scalar values using nano::from_string<tscalar>
        ///
        template <typename tscalar>
        auto collect() const
        {
            std::vector<std::pair<size_t, tscalar>> values;
            if (m_type == row_t::mode::data)
            {
                size_t col = 0;
                for (const auto& cell : m_cells)
                {
                    try
                    {
                        const auto value = ::nano::from_string<tscalar>(cell.m_data);
                        for (size_t span = 0; span < cell.m_span; ++ span)
                        {
                            values.emplace_back(col + span, value);
                        }
                    }
                    catch (std::exception&)
                    {
                    }
                    col += cell.m_span;
                }
            }
            return values;
        }

        ///
        /// \brief select the columns that satisfy the given operator
        ///
        template <typename tscalar, typename toperator>
        auto select(const toperator& op) const
        {
            std::vector<size_t> indices;
            // fixme: this is not very efficient because we iterate twice through the cells!
            for (const auto& cv : collect<tscalar>())
            {
                if (op(cv.second))
                {
                    indices.emplace_back(cv.first);
                }
            }
            return indices;
        }

        ///
        /// \brief access functions
        ///
        auto cols() const { return m_cols; }
        auto type() const { return m_type; }
        const auto& cells() const { return m_cells; }
        auto data(const size_t col) const { const auto* cell = find(col); assert(cell); return cell->m_data; }
        auto mark(const size_t col) const { const auto* cell = find(col); assert(cell); return cell->m_mark; }

    private:

        // attributes
        mode                m_type{mode::data};     ///< row type
        size_t              m_cols{0};              ///< current number of columns taking into account column spanning
        char                m_colfill{' '};         ///< current cell fill character
        size_t              m_colspan{1};           ///< current cell column span
        alignment           m_alignment{alignment::left};///< current cell alignment
        std::vector<cell_t> m_cells;                ///<
    };

    inline bool operator==(const row_t& r1, const row_t& r2)
    {
        return r1.type() == r2.type() && std::operator==(r1.cells(), r2.cells());
    }

    ///
    /// \brief stores and formats tabular data for display.
    ///
    class table_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        table_t() = default;

        ///
        /// \brief remove all rows, but keeps the header
        ///
        void clear();

        ///
        /// \brief append a row as a header, as a data or as a delimeter row
        ///
        row_t& delim() { m_rows.emplace_back(row_t::mode::delim); return *m_rows.rbegin(); }
        row_t& header() { m_rows.emplace_back(row_t::mode::header); return *m_rows.rbegin(); }
        row_t& append() { m_rows.emplace_back(row_t::mode::data); return *m_rows.rbegin(); }

        ///
        /// \brief (stable) sort the table using the given operator and columns
        /// e.g. toperator can be nano::make_[less|greater]_from_string<tscalar>
        ///
        template <typename toperator>
        void sort(const toperator& comp, const std::vector<size_t>& columns)
        {
            std::stable_sort(m_rows.begin(), m_rows.end(), [&] (const row_t& row1, const row_t& row2)
            {
                if (row1.type() == row_t::mode::data && row2.type() == row_t::mode::data)
                {
                    assert(row1.cols() == row2.cols());
                    for (const auto col : columns)
                    {
                        const auto* cell1 = row1.find(col);
                        const auto* cell2 = row2.find(col);
                        assert(cell1 && cell2);

                        if (comp(cell1->m_data, cell2->m_data))
                        {
                            return true;
                        }
                        else if (comp(cell2->m_data, cell1->m_data))
                        {
                            return false;
                        }
                    }
                    return true;
                }
                else
                {
                    return false;
                }
            });
        }

        ///
        /// \brief mark row-wise the selected columns with the given operator
        ///
        template <typename tmarker>
        void mark(const tmarker& marker, const char* marker_string = " (*)")
        {
            for (auto& row : m_rows)
            {
                for (const auto col : marker(row))
                {
                    row.mark(col, marker_string);
                }
            }
        }

        ///
        /// \brief access functions
        ///
        size_t cols() const;
        auto rows() const { return m_rows.size(); }
        const auto& content() const { return m_rows; }
        auto& row(const size_t r) { assert(r < rows()); return m_rows[r]; }
        const auto& row(const size_t r) const { assert(r < rows()); return m_rows[r]; }

    private:

        // attributes
        std::vector<row_t>      m_rows;
    };

    inline size_t table_t::cols() const
    {
        const auto op = [] (const row_t& row1, const row_t& row2) { return row1.cols() < row2.cols(); };
        const auto it = std::max_element(m_rows.begin(), m_rows.end(), op);
        return (it == m_rows.end()) ? size_t(0) : it->cols();
    }

    inline bool operator==(const table_t& t1, const table_t& t2)
    {
        return std::operator==(t1.content(), t2.content());
    }

    ///
    /// \brief pretty-print the table.
    ///
    NANO_PUBLIC std::ostream& operator<<(std::ostream& os, const table_t& table);

    ///
    /// \brief construct an operator to compare two strings numerically.
    ///
    template <typename tscalar>
    auto make_less_from_string()
    {
        return [] (const string_t& v1, const string_t& v2)
        {
            return  from_string<tscalar>(v1, std::numeric_limits<tscalar>::lowest()) <
                    from_string<tscalar>(v2, std::numeric_limits<tscalar>::max());
        };
    }

    ///
    /// \brief construct an operator to compare two strings numerically.
    ///
    template <typename tscalar>
    auto make_greater_from_string()
    {
        return [] (const string_t& v1, const string_t& v2)
        {
            return  from_string<tscalar>(v1, std::numeric_limits<tscalar>::max()) >
                    from_string<tscalar>(v2, std::numeric_limits<tscalar>::lowest());
        };
    }

    namespace detail
    {
        template <typename tscalar>
        auto min_element(const std::vector<std::pair<size_t, tscalar>>& values)
        {
            const auto comp = [] (const auto& cv1, const auto& cv2) { return cv1.second < cv2.second; };
            return std::min_element(values.begin(), values.end(), comp);
        }

        template <typename tscalar>
        auto max_element(const std::vector<std::pair<size_t, tscalar>>& values)
        {
            const auto comp = [] (const auto& cv1, const auto& cv2) { return cv1.second < cv2.second; };
            return std::max_element(values.begin(), values.end(), comp);
        }

        template <typename tscalar, typename toperator>
        std::vector<size_t> filter(const std::vector<std::pair<size_t, tscalar>>& values, const toperator& op)
        {
            std::vector<size_t> indices;
            std::for_each(values.begin(), values.end(), [&] (const auto& cv)
            {
                if (op(cv.second))
                {
                    indices.emplace_back(cv.first);
                }
            });
            return indices;
        }

        template <typename tscalar>
        std::vector<size_t> filter_less(const std::vector<std::pair<size_t, tscalar>>& values, const tscalar threshold)
        {
            return filter(values, [threshold = threshold] (const auto& cvlt) { return cvlt < threshold; });
        }

        template <typename tscalar>
        std::vector<size_t> filter_greater(const std::vector<std::pair<size_t, tscalar>>& values, const tscalar threshold)
        {
            return filter(values, [threshold = threshold] (const auto& cvgt) { return cvgt > threshold; });
        }
    }

    ///
    /// \brief select the column with the minimum value
    ///
    template <typename tscalar>
    auto make_marker_minimum_col()
    {
        return [=] (const row_t& row)
        {
            const auto values = row.collect<tscalar>();
            const auto it = detail::min_element(values);
            return (it == values.end()) ? std::vector<size_t>{} : std::vector<size_t>{it->first};
        };
    }

    ///
    /// \brief select the column with the maximum value
    ///
    template <typename tscalar>
    auto make_marker_maximum_col()
    {
        return [=] (const row_t& row) -> std::vector<size_t>
        {
            const auto values = row.collect<tscalar>();
            const auto it = detail::max_element(values);
            return (it == values.end()) ? std::vector<size_t>{} : std::vector<size_t>{it->first};
        };
    }

    ///
    /// \brief select the columns within [0, epsilon] from the maximum value
    ///
    template <typename tscalar>
    auto make_marker_maximum_epsilon_cols(const tscalar epsilon)
    {
        return [=] (const row_t& row)
        {
            const auto values = row.collect<tscalar>();
            const auto it = detail::max_element(values);
            return (it == values.end()) ? std::vector<size_t>{} : detail::filter_greater(values, it->second - epsilon);
        };
    }

    ///
    /// \brief select the columns within [0, epsilon] from the minimum value
    ///
    template <typename tscalar>
    auto make_marker_minimum_epsilon_cols(const tscalar epsilon)
    {
        return [=] (const row_t& row)
        {
            const auto values = row.collect<tscalar>();
            const auto it = detail::min_element(values);
            return (it == values.end()) ? std::vector<size_t>{} : detail::filter_less(values, it->second + epsilon);
        };
    }

    ///
    /// \brief select the columns within [0, percentage]% from the maximum value
    ///
    template <typename tscalar>
    auto make_marker_maximum_percentage_cols(const tscalar percentage)
    {
        return [=] (const row_t& row)
        {
            assert(percentage >= tscalar(1));
            assert(percentage <= tscalar(99));
            const auto values = row.collect<tscalar>();
            const auto it = detail::max_element(values);
            return  (it == values.end()) ? std::vector<size_t>{} : detail::filter_greater(values,
                    it->second - percentage * (it->second < 0 ? -it->second : +it->second) / tscalar(100));
        };
    }

    ///
    /// \brief select the columns within [0, percentage]% from the minimum value
    ///
    template <typename tscalar>
    auto make_marker_minimum_percentage_cols(const tscalar percentage)
    {
        return [=] (const row_t& row)
        {
            assert(percentage >= tscalar(1));
            assert(percentage <= tscalar(99));
            const auto values = row.collect<tscalar>();
            const auto it = detail::min_element(values);
            return  (it == values.end()) ? std::vector<size_t>{} : detail::filter_less(values,
                    it->second + percentage * (it->second < 0 ? -it->second : +it->second) / tscalar(100));
        };
    }
}
