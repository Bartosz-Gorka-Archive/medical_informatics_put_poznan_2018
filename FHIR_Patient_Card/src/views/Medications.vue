<template>
  <main class="l-app__main" role="main">

    <header class="c-toolbar">
      <div class="c-cell">
        <div class="c-cell__media">
          <i class="icon-footprint"></i>
        </div>
        <div class="c-cell__content">
          <h1 class="c-toolbar__title">Medications list</h1>
          <ol class="c-breadcrumb">
            <li class="c-breadcrumb__item">Medications</li>
          </ol>
        </div>
      </div>
    </header>

    <table class="table table--data">
      <thead>
        <tr>
          <th class="u-hiddenDown@md">No</th>
          <th>ID</th>
          <th>Code</th>
          <th>Display</th>
          <th>Status</th>
          <th>Is brand</th>
        </tr>
      </thead>
      <tfoot>
        <tr>
          <th class="u-hiddenDown@md">No</th>
          <th>ID</th>
          <th>Code</th>
          <th>Display</th>
          <th>Status</th>
          <th>Is brand</th>
        </tr>
      </tfoot>
      <tbody>
        <template v-for="(medication, index) in this.medications">
          <tr>
            <td data-label="No" class="u-hiddenDown@md">{{ index + 1 }}</td>
            <td data-label="ID">
              <router-link :to="{ name: 'single-medication', params: { medicationID: get(['resource', 'id'], medication) }}">
                {{ get(['resource', 'id'], medication) }}
              </router-link>
            </td>
            <td data-label="Code">{{ get(['resource', 'code', 'coding', 0, 'code'], medication) }}</td>
            <td data-label="Display">{{ get(['resource', 'code', 'coding', 0, 'display'], medication) }}</td>
            <td data-label="Status">{{ get(['resource', 'status'], medication) }}</td>
            <td data-label="Is brand">{{ get(['resource', 'isBrand'], medication) }}</td>
          </tr>
        </template>

      <infinite-loading
        v-if="loadingMedications"
        v-on:infinite="infiniteHandler"
        ref="infiniteLoading"
        spinner="bubbles">
      </infinite-loading>

      </tbody>
    </table>

  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters, mapActions } = createNamespacedHelpers('medication')

  export default {
    name: 'MedicationsView',
    computed: mapGetters(['loadingMedications', 'medications']),
    mounted () {
      mapActions(['clear'])
    },
    methods: {
      get (p, o) {
        return p.reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, o)
      },
      infiniteHandler (state) {
        this.$store.dispatch('medication/getMedications')
        .then(data => {
          mapGetters(['loadingMedications', 'medications'])
          if (this.loadingMedications) {
            state.loaded()
          } else {
            state.complete()
          }
        })
      }
    }
  }
</script>
