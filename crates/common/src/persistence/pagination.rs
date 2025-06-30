//! # Pagination Support
//!
//! Pagination structures and utilities for database queries.
//! Provides standard pagination types and enhanced utilities.

use serde::{Deserialize, Serialize};

/// Page request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRequest {
    pub page_number: u32,
    pub page_size: u32,
}

impl Default for PageRequest {
    fn default() -> Self {
        Self {
            page_number: 1,
            page_size: 50,
        }
    }
}

impl PageRequest {
    /// Create new page request
    pub fn new(page_number: u32, page_size: u32) -> Self {
        Self {
            page_number: page_number.max(1), // Ensure minimum page 1
            page_size: page_size.max(1),     // Ensure minimum size 1
        }
    }

    /// Convert to offset-based pagination
    pub fn to_pagination(&self) -> Pagination {
        let offset = if self.page_number > 0 {
            (self.page_number - 1) * self.page_size
        } else {
            0
        };
        Pagination::new(self.page_size, offset)
    }
}

/// Page result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub page_number: u32,
    pub page_size: u32,
    pub total_count: u64,
    pub total_pages: u32,
}

impl<T> Page<T> {
    /// Create new page from items and request
    pub fn new(items: Vec<T>, request: &PageRequest, total_count: u64) -> Self {
        let total_pages = if request.page_size == 0 {
            1
        } else {
            ((total_count as u32 + request.page_size - 1) / request.page_size).max(1)
        };

        Self {
            items,
            page_number: request.page_number,
            page_size: request.page_size,
            total_count,
            total_pages,
        }
    }

    /// Check if there's a next page
    pub fn has_next_page(&self) -> bool {
        self.page_number < self.total_pages
    }

    /// Check if there's a previous page
    pub fn has_previous_page(&self) -> bool {
        self.page_number > 1
    }
}

/// Pagination parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub limit: u32,
    pub offset: u32,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            limit: 50,
            offset: 0,
        }
    }
}

impl Pagination {
    /// Create new pagination with limit and offset
    pub fn new(limit: u32, offset: u32) -> Self {
        Self { limit, offset }
    }

    /// Create pagination for a specific page (1-indexed)
    pub fn page(page: u32, page_size: u32) -> Self {
        let offset = if page > 0 { (page - 1) * page_size } else { 0 };
        Self {
            limit: page_size,
            offset,
        }
    }

    /// Get the page number (1-indexed)
    pub fn page_number(&self) -> u32 {
        if self.limit == 0 {
            1
        } else {
            (self.offset / self.limit) + 1
        }
    }

    /// Get the next page pagination
    pub fn next_page(&self) -> Self {
        Self {
            limit: self.limit,
            offset: self.offset + self.limit,
        }
    }

    /// Get the previous page pagination
    pub fn previous_page(&self) -> Self {
        let offset = self.offset.saturating_sub(self.limit);
        Self {
            limit: self.limit,
            offset,
        }
    }
}

/// Paginated response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub total_count: u64,
    pub page_size: u32,
    pub page_offset: u32,
    pub has_more: bool,
}

impl<T> PaginatedResponse<T> {
    /// Create a new paginated response
    pub fn new(items: Vec<T>, total_count: u64, pagination: &Pagination) -> Self {
        let has_more = pagination.offset + pagination.limit < total_count as u32;

        Self {
            items,
            total_count,
            page_size: pagination.limit,
            page_offset: pagination.offset,
            has_more,
        }
    }

    /// Get the current page number (1-indexed)
    pub fn current_page(&self) -> u32 {
        if self.page_size == 0 {
            1
        } else {
            (self.page_offset / self.page_size) + 1
        }
    }

    /// Get the total number of pages
    pub fn total_pages(&self) -> u32 {
        if self.page_size == 0 {
            1
        } else {
            ((self.total_count as u32 + self.page_size - 1) / self.page_size).max(1)
        }
    }

    /// Check if there's a next page
    pub fn has_next_page(&self) -> bool {
        self.has_more
    }

    /// Check if there's a previous page
    pub fn has_previous_page(&self) -> bool {
        self.page_offset > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagination_creation() {
        let p1 = Pagination::new(10, 20);
        assert_eq!(p1.limit, 10);
        assert_eq!(p1.offset, 20);

        let p2 = Pagination::page(3, 10);
        assert_eq!(p2.limit, 10);
        assert_eq!(p2.offset, 20); // (3-1) * 10 = 20
    }

    #[test]
    fn test_pagination_page_number() {
        let p1 = Pagination::new(10, 0);
        assert_eq!(p1.page_number(), 1);

        let p2 = Pagination::new(10, 10);
        assert_eq!(p2.page_number(), 2);

        let p3 = Pagination::new(10, 25);
        assert_eq!(p3.page_number(), 3);
    }

    #[test]
    fn test_pagination_navigation() {
        let p1 = Pagination::new(10, 20);

        let next = p1.next_page();
        assert_eq!(next.limit, 10);
        assert_eq!(next.offset, 30);

        let prev = p1.previous_page();
        assert_eq!(prev.limit, 10);
        assert_eq!(prev.offset, 10);

        // Test boundary condition
        let p2 = Pagination::new(10, 5);
        let prev2 = p2.previous_page();
        assert_eq!(prev2.offset, 0);
    }

    #[test]
    fn test_paginated_response() {
        let items = vec![1, 2, 3, 4, 5];
        let pagination = Pagination::new(5, 10);
        let response = PaginatedResponse::new(items, 100, &pagination);

        assert_eq!(response.items.len(), 5);
        assert_eq!(response.total_count, 100);
        assert_eq!(response.page_size, 5);
        assert_eq!(response.page_offset, 10);
        assert_eq!(response.current_page(), 3);
        assert_eq!(response.total_pages(), 20);
        assert!(response.has_more);
        assert!(response.has_previous_page());
    }

    #[test]
    fn test_page_request_and_page() {
        // Test PageRequest
        let request = PageRequest::new(2, 10);
        assert_eq!(request.page_number, 2);
        assert_eq!(request.page_size, 10);

        // Test conversion to offset-based pagination
        let pagination = request.to_pagination();
        assert_eq!(pagination.limit, 10);
        assert_eq!(pagination.offset, 10); // (2-1) * 10

        // Test Page creation
        let items = vec!["item1", "item2", "item3"];
        let page = Page::new(items, &request, 25);

        assert_eq!(page.items.len(), 3);
        assert_eq!(page.page_number, 2);
        assert_eq!(page.page_size, 10);
        assert_eq!(page.total_count, 25);
        assert_eq!(page.total_pages, 3);
        assert!(page.has_previous_page());
        assert!(page.has_next_page());

        // Test first page
        let first_request = PageRequest::new(1, 10);
        let first_page = Page::new(vec!["item1"], &first_request, 25);
        assert!(!first_page.has_previous_page());
        assert!(first_page.has_next_page());

        // Test last page
        let last_request = PageRequest::new(3, 10);
        let last_page = Page::new(vec!["item1"], &last_request, 25);
        assert!(last_page.has_previous_page());
        assert!(!last_page.has_next_page());
    }

    #[test]
    fn test_page_request_bounds() {
        // Test minimum bounds
        let request = PageRequest::new(0, 0);
        assert_eq!(request.page_number, 1); // Should be at least 1
        assert_eq!(request.page_size, 1); // Should be at least 1

        // Test default
        let default_request = PageRequest::default();
        assert_eq!(default_request.page_number, 1);
        assert_eq!(default_request.page_size, 50);
    }
}
