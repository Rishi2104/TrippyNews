/**
 * TrippyNews / NewsCheck Infinite Scroll & Auto-Rendering
 *
 * Automatically fetches and appends subsequent pages as the user scrolls
 * near the bottom of the current feed, eliminating manual pagination buttons.
 */
(function () {
  'use strict';

  function initInfiniteScroll() {
    const sentinel = document.getElementById('infinite-scroll-sentinel');
    const feed = document.getElementById('stories-feed');
    const countDisplay = document.getElementById('stories-count-display');

    if (!sentinel || !feed) return;

    let isLoading = false;
    let nextUrl = sentinel.getAttribute('data-next-url');
    const totalStories = parseInt(sentinel.getAttribute('data-total-stories') || '0', 10);

    function setStatus(text, showSpinner) {
      const statusText = sentinel.querySelector('.scroll-status-text');
      const spinner = sentinel.querySelector('.spinner');
      if (statusText) statusText.textContent = text;
      if (spinner) spinner.style.display = showSpinner ? 'inline-block' : 'none';
    }

    function updateCount() {
      if (!countDisplay) return;
      const count = feed.querySelectorAll('.story-card').length;
      if (totalStories > 0) {
        countDisplay.textContent = `Showing 1–${count} of ${totalStories} stories`;
      } else {
        countDisplay.textContent = `${count} stories this run`;
      }
    }

    async function loadNext() {
      if (isLoading || !nextUrl) return;
      isLoading = true;
      setStatus('Loading more stories...', true);

      try {
        const res = await fetch(nextUrl);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const html = await res.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');

        const newCards = doc.querySelectorAll('#stories-feed .story-card');
        if (newCards.length === 0) {
          showFinished();
          return;
        }

        const fragment = document.createDocumentFragment();
        newCards.forEach(function (card) {
          card.classList.add('fade-in');
          fragment.appendChild(card);
        });
        feed.appendChild(fragment);

        updateCount();

        // Check next sentinel from the fetched document
        const newSentinel = doc.getElementById('infinite-scroll-sentinel');
        const upcomingUrl = newSentinel ? newSentinel.getAttribute('data-next-url') : null;

        if (upcomingUrl) {
          nextUrl = upcomingUrl;
          sentinel.setAttribute('data-next-url', nextUrl);
          isLoading = false;
          checkIntersection();
        } else {
          showFinished();
        }
      } catch (err) {
        console.error('Failed to auto-render next page:', err);
        setStatus('Scroll down to retry loading stories', false);
        isLoading = false;
      }
    }

    function showFinished() {
      isLoading = false;
      nextUrl = null;
      if (observer) {
        observer.disconnect();
      }
      const count = feed.querySelectorAll('.story-card').length;
      sentinel.innerHTML = '<div class="feed-end-notice"><span>✓ You\'re all caught up • All ' + count + ' stories loaded</span></div>';
    }

    let observer = null;
    if ('IntersectionObserver' in window) {
      observer = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              loadNext();
            }
          });
        },
        {
          rootMargin: '350px 0px',
          threshold: 0.05,
        }
      );
      observer.observe(sentinel);
    } else {
      window.addEventListener('scroll', function () {
        checkIntersection();
      }, { passive: true });
    }

    function checkIntersection() {
      if (isLoading || !nextUrl) return;
      const rect = sentinel.getBoundingClientRect();
      if (rect.top <= window.innerHeight + 350) {
        loadNext();
      }
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initInfiniteScroll);
  } else {
    initInfiniteScroll();
  }
})();
