<template>
  <div class="l-app">
    <div class="l-app__center">
      <div class="c-message">
        <div class="c-message__media">
          <h1 class="u-fs-128">404</h1>
        </div>
        <p class="c-message__title">Ooops, we have problem!</p>
        <p class="c-message__content">
          The page you are looking for doesn't exist.
        </p>
        <router-link to="/" class="btn btn--lg btn--smart btn--primary">Return to Home</router-link>
      </div>
    </div>
  </div>
</template>

<script>
  export default {
    name: 'NotFoundView'
  }
</script>
