<template>
  <main class="l-app__main" role="main">

    <header class="c-toolbar">
      <div class="c-cell">
        <div class="c-cell__media">
          <i class="icon-footprint"></i>
        </div>
        <div class="c-cell__content">
          <h1 class="c-toolbar__title">Observations list</h1>
          <ol class="c-breadcrumb">
            <li class="c-breadcrumb__item">Observations</li>
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
            <th>Value</th>
            <th>Subject</th>
          </tr>
        </thead>
        <tfoot>
          <tr>
            <th class="u-hiddenDown@md">No</th>
            <th>ID</th>
            <th>Code</th>
            <th>Display</th>
            <th>Status</th>
            <th>Value</th>
            <th>Subject</th>
          </tr>
        </tfoot>
        <tbody>
          <template v-for="(observation, index) in this.observations">
            <tr>
              <td data-label="No" class="u-hiddenDown@md">{{ index + 1 }}</td>
              <td data-label="ID">{{ get(['resource', 'id'], observation) }}</td>
              <td data-label="Code">{{ get(['resource', 'code', 'coding', 0, 'code'], observation) }}</td>
              <td data-label="Display">{{ get(['resource', 'code', 'coding', 0, 'display'], observation) }}</td>
              <td data-label="Status">{{ get(['resource', 'status'], observation) }}</td>
              <td data-label="Value">{{ get(['resource', 'valueQuantity', 'value'], observation) }} {{ get(['resource', 'valueQuantity', 'unit'], observation) }}</td>
              <td data-label="Subject" v-if="get(['resource', 'subject', 'reference'], observation)">
                <router-link :to="{ name: 'single-patient', params: { patientID: get(['resource', 'subject', 'reference'], observation) }}">{{ get(['resource', 'subject', 'reference'], observation) }}</router-link>
              </td>
              <td data-label="Subject" v-else></td>
            </tr>
          </template>

        <infinite-loading
          v-if="loadingObservations"
          v-on:infinite="infiniteHandler"
          ref="infiniteLoading"
          spinner="bubbles">
        </infinite-loading>

        </tbody>
      </table>
    </div>

  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters, mapActions } = createNamespacedHelpers('observation')

  export default {
    name: 'ObservationsView',
    computed: mapGetters(['loadingObservations', 'observations']),
    mounted () {
      mapActions(['clear'])
    },
    methods: {
      get (p, o) {
        return p.reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, o)
      },
      infiniteHandler (state) {
        this.$store.dispatch('observation/getObservations')
        .then(data => {
          mapGetters(['loadingObservations', 'observations'])
          if (this.loadingObservations) {
            state.loaded()
          } else {
            state.complete()
          }
        })
      }
    }
  }
</script>
