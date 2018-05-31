<template>
  <main class="l-app__main" role="main">

    <header class="c-toolbar">
      <div class="c-cell">
        <div class="c-cell__media">
          <i class="icon-footprint"></i>
        </div>
        <div class="c-cell__content">
          <h1 class="c-toolbar__title">Observation {{ get(['id'], selectedObservation)}}</h1>
          <ol class="c-breadcrumb">
            <li class="c-breadcrumb__item">
              <router-link :to="{ name: 'observations' }" class="c-breadcrumb__link">Observations</router-link>
            </li>
            <li class="c-breadcrumb__item">Observation {{ get(['id'], selectedObservation)}}</li>
          </ol>
        </div>
      </div>
    </header>

    <table class="table table--data">
      <thead>
        <tr>
          <th>Key</th>
          <th>Value</th>
        </tr>
      </thead>
      <tfoot>
        <tr>
          <th>Key</th>
          <th>Value</th>
        </tr>
      </tfoot>
      <tbody>
        <tr v-if="get(['id'], selectedObservation)">
          <td data-label="Key">ID</td>
          <td data-label="Value">{{ get(['id'], selectedObservation) }}</td>
        </tr>

        <tr v-if="get(['status'], selectedObservation)">
          <td data-label="Key">Status</td>
          <td data-label="Value">{{ get(['status'], selectedObservation) }}</td>
        </tr>

        <tr v-if="get(['meta', 'versionId'], selectedObservation)">
          <td data-label="Key">Version ID</td>
          <td data-label="Value">
            Current version {{ get(['meta', 'versionId'], selectedObservation) }}
            <template v-for="num in this.totalVersions - 1">
              <router-link :to="{ name: 'single-versioned-observation', params: { observationID: get(['id'], selectedObservation), versionNumber: num }}">
                [version {{ num }}]
              </router-link>
            </template>
          </td>
        </tr>

        <tr v-if="get(['meta', 'lastUpdated'], selectedObservation)">
          <td data-label="Key">Last updated</td>
          <td data-label="Value">{{ new Date(get(['meta', 'lastUpdated'], selectedObservation)).toLocaleString() }}</td>
        </tr>

        <tr v-if="get(['code', 'coding', 0, 'code'], selectedObservation)">
          <td data-label="Key">Code</td>
          <td data-label="Value">{{ get(['code', 'coding', 0, 'code'], selectedObservation) }}</td>
        </tr>

        <tr v-if="get(['code', 'coding', 0, 'display'], selectedObservation)">
          <td data-label="Key">Display</td>
          <td data-label="Value">{{ get(['code', 'coding', 0, 'display'], selectedObservation) }}</td>
        </tr>

        <tr v-if="get(['valueQuantity', 'value'], selectedObservation)">
          <td data-label="Key">Value</td>
          <td data-label="Value">{{ get(['valueQuantity', 'value'], selectedObservation) }}</td>
        </tr>

        <tr v-if="get(['valueQuantity', 'unit'], selectedObservation)">
          <td data-label="Key">Unit</td>
          <td data-label="Value">{{ get(['valueQuantity', 'unit'], selectedObservation) }}</td>
        </tr>

        <tr v-if="get(['valueQuantity', 'code'], selectedObservation)">
          <td data-label="Key">Value Quantity Code</td>
          <td data-label="Value">{{ get(['valueQuantity', 'code'], selectedObservation) }}</td>
        </tr>

        <tr v-if="get(['subject', 'reference'], selectedObservation)">
          <td data-label="Key">Subject</td>
          <td data-label="Value">
            <router-link :to="{ name: 'single-patient', params: { patientID: get(['subject', 'reference'], selectedObservation) }}">
              {{ get(['subject', 'reference'], selectedObservation) }}
            </router-link>
          </td>
        </tr>
      </tbody>
    </table>

    <div class="divider u-mt-20"><span>Record edition</span></div>

    <div class="u-mt-20 l-row">
      <div class="l-col-4@md"></div>
      <div class="l-col-4@md">
        <input type="text" class="form-input" placeholder="Code" v-model="code">
        <input type="text" class="form-input" placeholder="Display text" v-model="display">
        <input type="text" class="form-input" placeholder="Value" v-model="value">

        <button class="btn btn--info u-mt-10" @click="sendUpdate()">Update record</button>
      </div>
      <div class="l-col-4@md"></div>
    </div>

  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters } = createNamespacedHelpers('observation')

  export default {
    name: 'SingleObservationView',
    computed: mapGetters(['selectedObservation', 'totalVersions']),
    data () {
      return {
        display: '-',
        code: '-',
        value: '-'
      }
    },
    mounted () {
      this.observationID = this.$route.params.observationID
      this.$store.dispatch('observation/getSingleObservation', this.observationID)
      .then(data => {
        mapGetters(['selectedObservation', 'totalVersions'])
        this.display = ['code', 'coding', 0, 'display'].reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, this.selectedObservation)
        this.code = ['code', 'coding', 0, 'code'].reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, this.selectedObservation)
        this.value = ['valueQuantity', 'value'].reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, this.selectedObservation)
      })
    },
    methods: {
      sendUpdate () {
        this.$store.dispatch('observation/updateObservation', {
          code: this.code,
          observationID: this.observationID,
          display: this.display,
          value: this.value
        })
      },
      get (p, o) {
        return p.reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, o)
      }
    }
  }
</script>
